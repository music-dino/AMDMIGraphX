/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/matcher.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/gpu/device_name.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct ck_gemm
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 2)
            MIGRAPHX_THROW(name() + ": should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        auto r = op.compute_shape({a, b});
        if(mods.empty())
            return r;
        return r.with_type(mods.front()->get_output_shapes().front().type());
    }

    static bool is_ck_supported_type(shape::type_t t)
    {
        return contains({shape::half_type, shape::int8_type, shape::int32_type}, t);
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

struct ck_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(ck_gemm_softmax_gemm);

struct ck_fmha_fwd : fmha_fwd_base
{
    bool has_bias = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.scale, "scale"), f(self.has_bias, "has_bias"));
    }

    std::string name() const { return "gpu::ck_fmha_fwd"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        std::size_t expected = has_bias ? 4 : 3;
        if(inputs.size() > expected)
            inputs.erase(inputs.begin() + 2);
        return fmha_fwd_base::compute_shape(inputs, mods);
    }
};
MIGRAPHX_REGISTER_OP(ck_fmha_fwd);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot" and ins->name() != "quant_dot")
        return false;
    if(not ck_gemm::is_ck_supported_type(ins->get_shape().type()))
        return false;
    auto a          = ins->inputs().front()->get_shape();
    auto b          = ins->inputs().back()->get_shape();
    auto m          = a.lens()[a.lens().size() - 2];
    auto n          = b.lens().back();
    auto k          = a.lens().back();
    auto batch_size = std::accumulate(
        a.lens().rbegin() + 2, a.lens().rend(), std::size_t{1}, std::multiplies<std::size_t>());
    // Integer gemms must be divisible by 4 in ck
    if(contains({shape::int8_type, shape::int32_type}, ins->get_shape().type()))
    {
        if(m % 4 != 0)
            return false;
        if(n % 4 != 0)
            return false;
        if(k % 4 != 0)
            return false;
    }
    auto device_name = trim(split_string(get_device_name(), ':').front());
    if(starts_with(device_name, "gfx94"))
    {
        if(ins->get_shape().type() == shape::half_type)
        {
            if(batch_size >= 64)
                return m < 2048 or k <= 64 or n <= 384 or n >= 2048;
            return true;
        }
        return true;
    }
    return k <= 1024;
}

struct find_ck_gemm_pointwise
{
    // Find a gemm followed by a pointwise operation.
    auto matcher() const
    {
        auto gemm = match::skip(match::name("contiguous"))(
            match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm")));
        return match::name("pointwise")(match::any_of[match::inputs()](gemm.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto* pm      = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        auto inputs   = ins->inputs();
        auto gemm_it  = std::find(inputs.begin(), inputs.end(), x_ins);
        auto gemm_idx = gemm_it - inputs.begin();
        if(gemm_ins->get_shape().type() != shape::int32_type and
           ins->get_shape().type() != gemm_ins->get_shape().type())
            return;
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto input) {
               return not ck_gemm::is_ck_supported_type(input->get_shape().type());
           }))
            return;
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto input) {
               return not input->inputs().empty() and input->inputs().front()->name() == "capture";
           }))
            return;
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto input) {
               return not input->inputs().empty() and input->inputs().front()->name() == "capture";
           }))
            return;
        assert(gemm_it != inputs.end());
        if(gemm_idx != 0)
        {
            auto first_param    = pm->get_parameter(names[0]);
            auto gemm_param     = pm->get_parameter(names[gemm_idx]);
            auto new_gemm_param = pm->add_parameter(names[0] + "_0", gemm_param->get_shape());
            auto new_first_param =
                pm->add_parameter(names[gemm_idx] + "_0", first_param->get_shape());
            pm->replace_instruction(gemm_param, new_gemm_param);
            pm->replace_instruction(first_param, new_first_param);
            pm->remove_instruction(first_param);
            pm->remove_instruction(gemm_param);
        }
        inputs.erase(gemm_it);
        inputs.insert(inputs.begin(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());

        mpm.get_module().replace_instruction(ins, ck_gemm{gemm_ins->get_operator()}, inputs, {pm});
    }
};

struct find_ck_gemm
{
    auto matcher() const { return match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm{ins->get_operator()}, ins->inputs());
    }
};

struct find_ck_gemm_softmax_gemm
{
    auto matcher() const { return match::name("gpu::pre_gemm_softmax_gemm"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto v   = ins->get_operator().to_value();
        assert(v.contains("scale"));
        auto scale = v.at("scale").to<float>();
        mpm.get_module().replace_instruction(
            ins, ck_gemm_softmax_gemm{migraphx::make_op("dot"), scale}, ins->inputs());
    }
};

struct find_ck_fmha_fwd
{
    auto matcher() const { return match::name("gpu::pre_ck_fmha_fwd"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto v   = ins->get_operator().to_value();
        assert(v.contains("scale"));
        auto scale = v.at("scale").to<float>();
        bool bias = ins->inputs().size() == 4;
        mpm.get_module().replace_instruction(ins, ck_fmha_fwd{scale, bias}, ins->inputs());
    }
};

struct decomposed_fmha_checker
{
    bool matched              = false;
    instruction_ref q_param   = {};
    instruction_ref k_param   = {};
    instruction_ref v_param   = {};
    instruction_ref scale_input = {};
    instruction_ref bias_input  = {};
    bool has_scale = false;
    bool has_bias  = false;

    auto matcher() const
    {
        auto gemm1 =
            match::skip(match::name("convert"))(match::name("dot").bind("gemm1"));
        auto mul = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::any().bind("scale_input"), gemm1));
        auto add_with_scale = match::name("add")(
            match::nargs(2),
            match::either_arg(0, 1)(match::none_of(mul).bind("bias_input"), mul));
        auto add_no_scale = match::name("add")(
            match::nargs(2),
            match::either_arg(0, 1)(match::none_of(gemm1).bind("bias_input"), gemm1));
        auto softmax = match::skip(match::name("convert"))(
            match::softmax_input(match::any_of(add_with_scale, mul, add_no_scale, gemm1)));

        return match::name("dot")(match::arg(0)(softmax)).bind("gemm2");
    }

    void apply(module&, const match::matcher_result& r)
    {
        auto gemm1 = r.instructions["gemm1"];
        auto gemm2 = r.instructions["gemm2"];
        matched = true;
        q_param = gemm1->inputs()[0];
        k_param = gemm1->inputs()[1];
        v_param = gemm2->inputs()[1];
        if(contains(r.instructions, "scale_input"))
        {
            has_scale  = true;
            scale_input = r.instructions["scale_input"];
        }
        if(contains(r.instructions, "bias_input"))
        {
            has_bias  = true;
            bias_input = r.instructions["bias_input"];
        }
    }
};

static instruction_ref trace_to_param(instruction_ref ins)
{
    while(ins->name() != "@param")
    {
        if(ins->inputs().empty())
            return {};
        ins = ins->inputs()[0];
    }
    return ins;
}

struct find_ck_fmha_attention
{
    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto group_ins = r.instructions["group"];
        auto* submod   = group_ins->module_inputs().front();

        if(group_ins->module_inputs().size() > 1)
            return;

        auto return_ins = std::prev(submod->end());
        if(return_ins->name() != "@return" or return_ins->inputs().size() > 1)
            return;

        decomposed_fmha_checker checker;
        match::find_matches(*submod, checker);

        if(not checker.matched)
            return;

        if(checker.q_param->name() != "@param" or checker.k_param->name() != "@param" or
           checker.v_param->name() != "@param")
            return;

        auto group_inputs = group_ins->inputs();
        auto param_map    = submod->get_ins_param_map(group_inputs, true);

        auto q_main = param_map.at(checker.q_param);
        auto k_main = param_map.at(checker.k_param);
        auto v_main = param_map.at(checker.v_param);

        if(not ck_gemm::is_ck_supported_type(q_main->get_shape().type()))
            return;
        if(q_main->get_shape().ndim() != 4)
            return;
        if(q_main->get_shape().strides().back() != 1)
            return;

        float scale = 1.0f;
        instruction_ref scale_main = {};
        if(checker.has_scale)
        {
            auto scale_param = trace_to_param(checker.scale_input);
            if(scale_param->name() != "@param")
                return;
            scale_main = param_map.at(scale_param);
            if(not scale_main->can_eval())
                return;
            scale_main->eval().visit([&](const auto s) {
                if(not std::all_of(
                       s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                    return;
                scale = s.front();
            });
        }

        instruction_ref bias_main = {};
        if(checker.has_bias)
        {
            auto bias_param = trace_to_param(checker.bias_input);
            if(bias_param->name() != "@param")
                return;
            bias_main = param_map.at(bias_param);
        }

        module ck_mod;
        std::size_t param_idx = 0;
        auto ck_q = ck_mod.add_parameter(param_name(param_idx++), q_main->get_shape());
        auto ck_k = ck_mod.add_parameter(param_name(param_idx++), k_main->get_shape());

        std::vector<instruction_ref> ck_inputs = {ck_q, ck_k};

        if(checker.has_scale)
        {
            auto ck_scale =
                ck_mod.add_parameter(param_name(param_idx++), scale_main->get_shape());
            ck_inputs.push_back(ck_scale);
        }

        if(checker.has_bias)
        {
            auto ck_bias =
                ck_mod.add_parameter(param_name(param_idx++), bias_main->get_shape());
            ck_inputs.push_back(ck_bias);
        }

        auto ck_v = ck_mod.add_parameter(param_name(param_idx++), v_main->get_shape());
        ck_inputs.push_back(ck_v);

        auto ck_fmha_ins =
            ck_mod.add_instruction(ck_fmha_fwd{scale, checker.has_bias}, ck_inputs);
        ck_mod.add_return({ck_fmha_ins});

        auto ck_mod_ref = mpm.create_module("ck_" + submod->name(), std::move(ck_mod));
        ck_mod_ref->set_bypass();

        mpm.get_module().replace_instruction(
            group_ins,
            make_op("group", {{"tag", "attention"}}),
            group_ins->inputs(),
            {submod, ck_mod_ref});
    }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_ck_fmha_attention{});
    match::find_matches(mpm, find_ck_fmha_fwd{});
    match::find_matches(mpm, find_ck_gemm_softmax_gemm{});
    match::find_matches(mpm, find_ck_gemm_pointwise{});
    match::find_matches(mpm, find_ck_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

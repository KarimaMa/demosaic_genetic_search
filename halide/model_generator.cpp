#include "Halide.h"

#include <fstream>
#include <iostream>

#include "decode_base64.h"

using std::string;
using std::vector;

namespace {

using namespace Halide;

Var x{"x"}, y{"y"}, n{"n"}, ci{"ci"}, co{"co"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

struct Op {
    // Fields populated by parse()
    int id = 0;
    string name;
    int out_channels = 0;
    int in_channels[3] = {};
    int groups = 0;
    int inputs[3] = {};
    int scale_factor = 0;

    int weight0_shape[4] = {};
    string weight0_b64;
    int weight1_shape[4] = {};
    string weight1_b64;

    // Fields populated by define()
    Buffer<float> weights0, weights1;
    Func func;
    bool pointwise = true;
    bool may_inline = true;
    bool must_inline = false;
    bool unroll_channels = false;
    vector<int> outputs;
    bool non_power_of_two = false;

    // Fields set by schedule()
    Func compute_at;

    void parse(std::ifstream &stream) {
        stream >> id
               >> name
               >> out_channels
               >> in_channels[0]
               >> in_channels[1]
               >> in_channels[2]
               >> groups
               >> inputs[0]
               >> inputs[1]
               >> inputs[2]
               >> scale_factor
               >> weight0_shape[0]
               >> weight0_shape[1]
               >> weight0_shape[2]
               >> weight0_shape[3]
               >> weight0_b64
               >> weight1_shape[0]
               >> weight1_shape[1]
               >> weight1_shape[2]
               >> weight1_shape[3]
               >> weight1_b64;
    }

    Buffer<float> decode_buffer(const int *shape, const string &name, const string &buffer) {
        string decoded = base64_decode(buffer);
        Buffer<float> buf({shape[3], shape[2], shape[1], shape[0]}, name);
        assert(decoded.size() == buf.size_in_bytes());
        memcpy((void *)buf.data(), (const void *)(&decoded[0]), buf.size_in_bytes());
        return buf;
    }

    Expr to_float(Expr e) {
        return cast<float>(e);
    }

    void unroll_in_space(Func f, int factor) {
        Var xu, yu;
        f.align_bounds(x, factor)
            .align_bounds(y, factor)
            .tile(x, y, xu, yu, factor, factor, TailStrategy::RoundUp)
            .unroll(xu)
            .unroll(yu);

    }

    void define(vector<Op> &ops, const GeneratorInput<Buffer<float>> &raw, LoopLevel tiles) {
        func = Func(name + "_" + std::to_string(id));

        Func in_1, in_2, in_3;
        if (inputs[0] >= 0) {
            in_1 = ops[inputs[0]].func;
        }
        if (inputs[1] >= 0) {
            in_2 = ops[inputs[1]].func;
        }
        if (inputs[2] >= 0) {
            in_3 = ops[inputs[2]].func;
        }

        if (name == "Add") {
            if (in_channels[0] > in_channels[1]) {
                int r = in_channels[0] / in_channels[1];
                func(x, y, co) = in_1(x, y, co) + in_2(x, y, co / r);
            } else if (in_channels[0] < in_channels[1]) {
                int r = in_channels[1] / in_channels[0];
                func(x, y, co) = in_1(x, y, co / r) + in_2(x, y, co);
            } else {
                func(x, y, co) = in_1(x, y, co) + in_2(x, y, co);
            }
        } else if (name == "Conv1D") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights0", weight0_b64);
            weights1 = decode_buffer(weight1_shape, name + "." + std::to_string(id) + ".weights1", weight1_b64);
            int num_v_filters = out_channels / 2;

            assert(weight0_shape[3] == 1);
            assert(weight1_shape[2] == 1);

            int in_group_size = in_channels[0] / groups;
            int out_group_size = num_v_filters / groups;
            Expr group = min(co, num_v_filters - 1) / out_group_size;

            Expr v = 0.0f;
            RDom rv(0, weight0_shape[2]);
            for (int ci = 0; ci < in_group_size; ci++) {
                v += weights0(0, rv, ci, min(co, num_v_filters - 1)) * in_1(x, y + rv - weight0_shape[2]/2, ci + group * in_group_size);
            }

            RDom rh(0, weight1_shape[3]);
            group = 0; //max(co - num_v_filters, 0) / out_group_size;
            Expr h = 0.0f;
            for (int ci = 0; ci < in_group_size; ci++) {
                h += weights1(rh, 0, ci, max(co - num_v_filters, 0)) * in_1(x + rh - weight1_shape[3]/2, y, ci + group * in_group_size);
            }

            func(x, y, co) = select(co < num_v_filters, sum(v), sum(h));

            pointwise = false;
            unroll_channels = true;

        } else if (name == "Conv1x1") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights", weight0_b64);
            assert(weights0.dim(0).extent() == 1);
            assert(weights0.dim(1).extent() == 1);
            assert(weights0.dim(2).extent() == in_channels[0] / groups);
            assert(weights0.dim(3).extent() == out_channels);
            Expr e = 0.0f;
            int in_group_size = in_channels[0] / groups;
            int out_group_size = out_channels / groups;
            Expr group = co / out_group_size;
            for (int ci = 0; ci < in_group_size; ci++) {
                e += weights0(0, 0, ci, co) * in_1(x, y, ci + group * in_group_size);
            }

            if (outputs.size() == 1 && ops[outputs[0]].name == "Relu") {
                func(x, y, co) = max(e, 0);
                ops[outputs[0]].name = "Id";
            } else {
                func(x, y, co) = e;
            }

            pointwise = false;

        } else if (name == "Conv2D") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights", weight0_b64);
            assert(weights0.dim(0).extent() == 3);
            assert(weights0.dim(1).extent() == 3);
            assert(weights0.dim(2).extent() == in_channels[0] / groups);
            assert(weights0.dim(3).extent() == out_channels);

            Expr e = 0.0f;
            int in_group_size = in_channels[0] / groups;
            int out_group_size = out_channels / groups;
            Expr group = co / out_group_size;
            RDom r(0, 3, 0, 3);
            for (int ci = 0; ci < in_group_size; ci++) {
                e += weights0(r.x, r.y, ci, co) * in_1(x + r.x - 1, y + r.y - 1, ci + group * in_group_size);
            }
            Func inner_sum("conv2d_sum");
            inner_sum(x, y, co) += e;

            inner_sum.compute_at(func, co)
                .vectorize(x)
                .unroll(co)
                .unroll(y)
                .update()
                .reorder(co, x, y, r.x, r.y)
                .vectorize(x)
                .unroll(y)
                .unroll(co);

            if (outputs.size() == 1 && ops[outputs[0]].name == "Relu") {
                func(x, y, co) = max(inner_sum(x, y, co), 0);
                ops[outputs[0]].name = "Id";
            } else {
                func(x, y, co) = inner_sum(x, y, co);
            }
            pointwise = false;
            may_inline = false;

            Var yu;
            func.split(y, y, yu, 4, TailStrategy::RoundUp)
                .reorder(yu, x, y, co)
                .unroll(yu);

        } else if (name == "LearnedDownsample") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights", weight0_b64);

            int in_group_size = in_channels[0] / groups;
            int out_group_size = out_channels / groups;
            Expr group = co / out_group_size;

            int kernel_size = weights0.dim(0).extent();
            int factor = kernel_size / 2;
            // For a factor of two, we have 4x4 kernels, and want to start at -1.
            // For a factor of 3, we have 7x7 kernels, and we want to start at -2.
            int margin = (kernel_size - factor) / 2;

            assert(weights0.dim(0).extent() == weights0.dim(1).extent());
            assert(weights0.dim(2).extent() * groups == in_channels[0]);
            assert(weights0.dim(3).extent() == out_channels);

            if (true) {
                Func interleave{"interleave"};
                Var u, v;
                interleave(x, y, u, co) = in_1(factor*x + u, y, co);

                Expr e = 0.0f;
                RDom r(0, kernel_size, 0, kernel_size);
                Func inner_sum("LearnedDownsample_sum");
                for (int ci = 0; ci < in_group_size; ci++) {
                    e += weights0(r.x, r.y, ci, co) *
                        interleave(x + (r.x - margin)/factor,
                                   factor*y + r.y - margin,
                                   (r.x - margin) % factor,
                                   ci + group * in_group_size);
                }
                inner_sum(x, y, co) += e;

                func(x, y, co) = inner_sum(x, y, co);

                interleave
                    .compute_at(tiles)
                    .store_in(MemoryType::Stack)
                    .align_storage(x, 8)
                    .reorder(u, co, x, y)
                    .unroll(u)
                    .unroll(co)
                    .vectorize(x, 8, TailStrategy::RoundUp);
                /*
                inner_sum.compute_at(func, co)
                    .reorder(x, y, co)
                    .unroll(y)
                    .vectorize(x)
                    .unroll(co)
                    .update()
                    .reorder(x, y, co, r.x, r.y)
                    .unroll(y)
                    .vectorize(x)
                    .unroll(co);
                */
            } else {
                non_power_of_two = true;

                Expr e = 0.0f;
                RDom r(0, kernel_size, 0, kernel_size);
                for (int ci = 0; ci < in_group_size; ci++) {
                    e += (weights0(r.x, r.y, ci, co) *
                          in_1(factor*x + r.x - margin,
                               factor*y + r.y - margin,
                               ci + group * in_group_size));
                }
                func(x, y, co) = sum(e, "LearnedDownsample_sum");
            }
            pointwise = false;
        } else if (name == "Flat2Quad") {
            func(x, y, co) = mux(co, {in_1(2*x, 2*y, 0),
                                      in_1(2*x + 1, 2*y, 0),
                                      in_1(2*x, 2*y + 1, 0),
                                      in_1(2*x + 1, 2*y + 1, 0)});

            pointwise = false;
            unroll_channels = true;
        } else if (name == "GreenExtractor") {
            assert(in_channels[0] == 4);
            assert(in_channels[1] == 2);
            // 1 output channel, 4 + 2 input channels, which are
            // presumably bayer raw + 2 predicted greens.
            func(x, y, co) = select(x % 2 == 0 && y % 2 == 0,
                                    in_1(x/2, y/2, 0),
                                    x % 2 == 1 && y % 2 == 0,
                                    in_2(x/2, y/2, 0),
                                    x % 2 == 0 && y % 2 == 1,
                                    in_2(x/2, y/2, 1),
                                    in_1(x/2, y/2, 3));
            pointwise = false;
            may_inline = false;
        } else if (name == "GreenRBExtractorOp") {
            // Real input.
            func(x, y, co) = mux(co, {in_1(2*x + 1, 2*y, 0),
                                      in_1(2*x, 2*y + 1, 0)});
            pointwise = false;
            unroll_channels = true;
        } else if (name == "GroupedSum") {
            // Sum in groups
            int values_per_group = in_channels[0] / out_channels;
            Expr e = 0.0f;
            for (int i = 0; i < values_per_group; i++) {
                e += in_1(x, y, co * values_per_group + i);
            }
            func(x, y, co) = e;
            pointwise = false;
        } else if (name == "InterleavedSum") {
            int values_per_group = in_channels[0] / out_channels;
            if (values_per_group == 1) {
                func(x, y, co) = in_1(x, y, co);
                must_inline = true;
            } else {
                Expr e = 0.0f;
                for (int i = 0; i < values_per_group; i++) {
                    e += in_1(x, y, co + i * out_channels);
                }
                func(x, y, co) = e;
                pointwise = false;
            }

        } else if (name == "Input(Mosaic)") {
            // Real input
            func(x, y, co) = to_float(mux(co, {raw(2*x, 2*y),
                                               raw(2*x + 1, 2*y),
                                               raw(2*x, 2*y + 1),
                                               raw(2*x + 1, 2*y + 1)}));
            pointwise = false;
            unroll_channels = true;
        } else if (name == "Input(GreenExtractor)") {
            func(x, y, co) = in_1(x, y, co);
            must_inline = true;
        } else if (name == "Input(Green@GrGb)") {
            // Real input.
            func(x, y, co) = to_float(mux(co, {raw(2*x, 2*y),
                                               raw(2*x + 1, 2*y + 1)}));
            pointwise = false;
            unroll_channels = true;
        } else if (name == "Input(GreenQuad)") {
            func(x, y, co) = in_1(x, y, co);
            must_inline = true;
        } else if (name == "Input(Green@RB)") {
            func(x, y, co) = in_1(x, y, co);
            must_inline = true;
        } else if (name == "Input(RBdiffG_GreenQuad)") {
            func(x, y, co) = in_1(x, y, co);
            must_inline = true;
        } else if (name == "Input(RedBlueBayer)") {
            // Real input.
            func(x, y, co) = to_float(mux(co, {raw(2*x + 1, 2*y),
                                               raw(2*x, 2*y + 1)}));
            pointwise = false;
            unroll_channels = true;
        } else if (name == "Mul") {
            if (in_channels[0] > in_channels[1]) {
                int r = in_channels[0] / in_channels[1];
                func(x, y, co) = in_1(x, y, co) * in_2(x, y, co / r);
            } else if (in_channels[0] < in_channels[1]) {
                int r = in_channels[1] / in_channels[0];
                func(x, y, co) = in_1(x, y, co / r) * in_2(x, y, co);
            } else {
                func(x, y, co) = in_1(x, y, co) * in_2(x, y, co);
            }
        } else if (name == "Pack") {
            int factor = std::sqrt(out_channels / in_channels[0]);
            if (in_channels[0] * factor * factor != out_channels) {
                std::cerr << "Bad pack factor: " << factor << "\n";
                assert(false);
            }
            vector<Expr> values(factor * factor);
            for (int i = 0; i < factor; i++) {
                for (int j = 0; j < factor; j++) {
                    // TODO: figure out row vs col major
                    values[i*factor + j] = in_1(x*factor + i, y*factor + j, co / (factor * factor));
                }
            }
            func(x, y, co) = mux(co % (factor * factor), values);
            pointwise = false;
            unroll_channels = true;
            must_inline = false;
        } else if (name == "Unpack") {
            int factor = std::sqrt(in_channels[0] / out_channels);
            if (out_channels * factor * factor != in_channels[0]) {
                std::cerr << "Bad unpack factor: " << factor << "\n";
                assert(false);
            }
            vector<Expr> values(factor * factor);
            for (int i = 0; i < factor; i++) {
                for (int j = 0; j < factor; j++) {
                    // TODO: figure out row vs col major
                    values[i*factor + j] = in_1(x/factor, y/factor, (co * factor + i) * factor + j);
                }
            }
            func(x, y, co) = mux((x % factor) * factor + (y % factor), values);
            //unroll_in_space(func, factor);
            pointwise = false;
            unroll_channels = true;
            may_inline = false;
        } else if (name == "Relu") {
            func(x, y, co) = max(0.0f, in_1(x, y, co));
        } else if (name == "Id" || name == "SGreenExtractor") {
            func(x, y, co) = in_1(x, y, co);
        } else if (name == "SRGBExtractor") {
            assert(in_channels[0] == 1 && in_channels[1] == 2);
            func(x, y, co) = mux(co, {in_2(x, y, 0), in_1(x, y, 0), in_2(x, y, 1)});
            unroll_channels = true;
        } else if (name == "RGBExtractor") {
            // Args are, 4 greens, 2 red/blue from the bayer, 6 predicted chroma
            /*
              # fill in reds
              img[:,0,0::2,0::2] = chromapred[:,0,:,:]
              img[:,0,0::2,1::2] = redbluebayer[:,0,:,:]
              img[:,0,1::2,0::2] = chromapred[:,2,:,:]
              img[:,0,1::2,1::2] = chromapred[:,3,:,:]

              # fill in greens
              img[:,1,:,:] = self.pixel_shuffle(fullgreen)[:,0,:,:]

              # fill in blues
              img[:,2,0::2,0::2] = chromapred[:,4,:,:]
              img[:,2,0::2,1::2] = chromapred[:,1,:,:]
              img[:,2,1::2,0::2] = redbluebayer[:,1,:,:]
              img[:,2,1::2,1::2] = chromapred[:,5,:,:]
            */

            func(x, y, co) = select(x % 2 == 0 && y % 2 == 0,
                                    mux(co, {in_3(x/2, y/2, 0), in_1(x/2, y/2, 0), in_3(x/2, y/2, 4)}),
                                    x % 2 == 1 && y % 2 == 0,
                                    mux(co, {in_2(x/2, y/2, 0), in_1(x/2, y/2, 1), in_3(x/2, y/2, 1)}),
                                    x % 2 == 0 && y % 2 == 1,
                                    mux(co, {in_3(x/2, y/2, 2), in_1(x/2, y/2, 2), in_2(x/2, y/2, 1)}),
                                    mux(co, {in_3(x/2, y/2, 3), in_1(x/2, y/2, 3), in_3(x/2, y/2, 5)}));
            pointwise = false;
            may_inline = false;
            unroll_channels = true;
        } else if (name == "Softmax") {
            Func max_arg{"max_arg"};
            Expr m = in_1(x, y, 0);
            for (int i = 1; i < in_channels[0]; i++) {
                m = max(m, in_1(x, y, i));
            }
            max_arg(x, y) = m;
            Func exp_args{"exp_args"};
            exp_args(x, y, co) = fast_exp(in_1(x, y, co) - max_arg(x, y));
            Expr sum = 0.0f;
            for (int i = 0; i < in_channels[0]; i++) {
                sum += exp_args(x, y, i);
            }
            Func inv_sum{"inv_sum"};
            inv_sum(x, y) = 1.0f / sum;
            func(x, y, co) = exp_args(x, y, co) * inv_sum(x, y);
            exp_args.compute_at(func, x).vectorize(x);
            inv_sum.compute_at(func, x).vectorize(x);
            max_arg.compute_at(func, x).vectorize(x);
            pointwise = false;
            may_inline = false;
        } else if (name == "Stack") {
            func(x, y, co) = select(co < in_channels[0],
                                    in_1(x, y, clamp(co, 0, in_channels[0] - 1)),
                                    in_2(x, y, clamp(co - in_channels[0], 0, in_channels[1] - 1)));
            unroll_channels = true;
            pointwise = false;
            must_inline = true;
        } else if (name == "Sub") {
            if (in_channels[0] > in_channels[1]) {
                int r = in_channels[0] / in_channels[1];
                func(x, y, co) = in_1(x, y, co) - in_2(x, y, co / r);
            } else if (in_channels[0] < in_channels[1]) {
                int r = in_channels[1] / in_channels[0];
                func(x, y, co) = in_1(x, y, co / r) - in_2(x, y, co);
            } else {
                func(x, y, co) = in_1(x, y, co) - in_2(x, y, co);
            }
        } else if (name == "BilinearUpsample") {
            // Bilinear
            if (scale_factor == 2) {
                Expr x_near = (x / 2) + (x % 2);
                Expr x_far = (x / 2) + 1 - (x % 2);
                Expr y_near = (y / 2) + (y % 2);
                Expr y_far = (y / 2) + 1 - (y % 2);
                func(x, y, co) =
                    (9 * in_1(x_near, y_near, co) +
                     3 * (in_1(x_near, y_far, co) + in_1(x_far, y_near, co)) +
                     in_1(x_far, y_far, co)) * (1.0f / 16);
            } else {
                Expr w_x = ((x % scale_factor)*2 + 1) / cast<float>(scale_factor * 2);
                Expr w_y = ((y % scale_factor)*2 + 1) / cast<float>(scale_factor * 2);
                Expr xi = x/scale_factor;
                Expr yi = y/scale_factor;
                func(x, y, co) =
                    lerp(
                         lerp(in_1(xi, yi, co),
                              in_1(xi + 1, yi, co),
                              w_x),
                         lerp(in_1(xi, yi + 1, co),
                              in_1(xi + 1, yi + 1, co),
                              w_x),
                         w_y);
            }
            unroll_in_space(func, scale_factor);
            pointwise = false;
            may_inline = false;
        } else if (name == "LearnedUpsample") {

            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights", weight0_b64);

            int factor = weights0.dim(0).extent();

            assert(weights0.dim(0).extent() == factor);
            assert(weights0.dim(1).extent() == factor);
            assert(weights0.dim(2).extent() == out_channels / groups);
            assert(weights0.dim(3).extent() == in_channels[0]);

            Expr e = 0.0f;
            int in_group_size = in_channels[0] / groups;
            int out_group_size = out_channels / groups;
            Expr group = co / out_group_size;
            for (int ci = 0; ci < in_group_size; ci++) {
                e += (weights0(x % factor, y % factor, co % out_group_size, ci + group * in_group_size) *
                      in_1(x / factor, y / factor, ci + group * in_group_size));
            }
            func(x, y, co) = e;
            pointwise = false;
            may_inline = false;
            unroll_in_space(func, factor);
        } else {
            std::cerr << "Unknown op: " << name << "\n";
            assert(false);
        }
    }

    void dump() {
        std::cout << id << " "
                  << name << " "
                  << out_channels << " "
                  << in_channels[0] << " "
                  << in_channels[1] << " "
                  << in_channels[2] << " "
                  << groups << " "
                  << inputs[0] << " "
                  << inputs[1] << " "
                  << inputs[2] << " "
                  << scale_factor << " "
                  << weight0_shape[0] << " "
                  << weight0_shape[1] << " "
                  << weight0_shape[2] << " "
                  << weight0_shape[3] << " "
                  << weight0_b64 << " "
                  << weight1_shape[0] << " "
                  << weight1_shape[1] << " "
                  << weight1_shape[2] << " "
                  << weight1_shape[3] << " "
                  << weight1_b64 << "\n";
    }

};

class Model : public Halide::Generator<Model> {
public:
    Input<Buffer<float>> raw{"raw", 2};
    GeneratorParam<string> model_file{"model_file", ""};
    Output<Buffer<float>> output{"output", 3};

    vector<Op> ops;

    void parse_model_file(const std::string &filename) {
        std::ifstream f;
        f.open(filename);
        while (1) {
            Op op;
            op.parse(f);
            if (f.good()) {
                ops.push_back(op);
            } else if (f.eof()) {
                break;
            } else {
                std::cerr << "Failed to parse\n";
                assert(false);
            }
        }
        f.close();

        // Connect up outputs
        for (auto &op : ops) {
            if (op.inputs[0] >= 0) {
                ops[op.inputs[0]].outputs.push_back(op.id);
            }
            if (op.inputs[1] >= 0) {
                ops[op.inputs[1]].outputs.push_back(op.id);
            }
            if (op.inputs[2] >= 0) {
                ops[op.inputs[2]].outputs.push_back(op.id);
            }
        }

        // Dedup input nodes
        std::map<string, int> inputs;
        for (auto &op : ops) {
            if (Internal::starts_with(op.name, "Input(") &&
                inputs.find(op.name) == inputs.end()) {
                inputs[op.name] = op.id;
            }
            for (int &i : op.inputs) {
                if (i >= 0) {
                    if (Internal::starts_with(ops[i].name, "Input(")) {
                        i = inputs[ops[i].name];
                    }
                }
            }
        }
    }

    void generate() {
        parse_model_file(model_file);
        LoopLevel tiles;
        for (Op &op: ops) {
            op.define(ops, raw, tiles);
        }
        output = ops.back().func;
        tiles.set(LoopLevel{output, xo});

        output
            .align_bounds(x, 16)
            .align_bounds(y, 16)
            .tile(x, y, xo, yo, x, y, 256 * 3, 240, TailStrategy::RoundUp)
            .reorder(co, x, y, xo, yo)
            .vectorize(x, 32, TailStrategy::RoundUp)
            .unroll(co);

        output.dim(0).set_min(0);
        output.dim(1).set_min(0);
        output.dim(2).set_bounds(0, 3);

        for (size_t i = 0; i < ops.size() - 1; i++)  {
            bool should_schedule = ops[i].outputs.size() > 1;
            if (ops[i].outputs.size() == 1) {
                should_schedule = !ops[ops[i].outputs[0]].pointwise;
            }
            should_schedule |= !ops[i].may_inline;
            should_schedule &= !ops[i].must_inline;
            if (should_schedule) {
                if (ops[i].non_power_of_two) {
                    ops[i].func
                        .store_in(MemoryType::Stack)
                        .compute_at(output, xo)
                        .reorder(co, x, y);
                } else {
                    ops[i].func
                        .store_in(MemoryType::Stack)
                        .align_storage(x, 8)
                        .compute_at(output, xo)
                        .vectorize(x, 32, TailStrategy::RoundUp)
                        .reorder(co, x, y);
                }
                if (ops[i].unroll_channels) {
                    ops[i].func.unroll(co);
                }

                // Not strictly necessary, but useful for debugging the definitions
                ops[i].func.bound(co, 0, ops[i].out_channels);
            } else {
                for (int o : ops[i].outputs) {
                    ops[o].unroll_channels |= ops[i].unroll_channels;
                    ops[o].pointwise &= ops[i].pointwise;
                }
            }
        }

        /*
        for (size_t i = 0; i < ops.size(); i++) {
            ops[i].func.compute_root().debug_to_file("debug_" + ops[i].func.name() + ".tmp");
        }
        */
    }
};

}

HALIDE_REGISTER_GENERATOR(Model, halide_model)

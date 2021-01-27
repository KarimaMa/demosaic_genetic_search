#include "Halide.h"

#include <fstream>
#include <iostream>

#include "decode_base64.h"

using std::string;
using std::vector;

namespace {

using namespace Halide;

Var x{"x"}, y{"y"}, n{"n"}, ci{"ci"}, co{"co"};

struct Op {
    // Fields populated by parse()
    int id = 0;
    string name;
    int out_channels = 0;
    int in_channels[3] = {};
    int groups = 0;
    int inputs[3] = {};

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

    void define(vector<Op> &ops, const GeneratorInput<Buffer<float>> &raw) {
        func = Func(name + "_" + std::to_string(id));

        Func in_1, in_2, in_3;
        if (inputs[0] >= 0) {
            in_1 = ops[inputs[0]].func;
            ops[inputs[0]].outputs.push_back(id);
        }
        if (inputs[1] >= 0) {
            in_2 = ops[inputs[1]].func;
            ops[inputs[1]].outputs.push_back(id);
        }
        if (inputs[2] >= 0) {
            in_3 = ops[inputs[2]].func;
            ops[inputs[2]].outputs.push_back(id);
        }

        if (name == "Add") {
            func(x, y, co) = in_1(x, y, co) + in_2(x, y, co);
        } else if (name == "Conv1D") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights0", weight0_b64);
            weights1 = decode_buffer(weight1_shape, name + "." + std::to_string(id) + ".weights1", weight1_b64);
            int num_v_filters = out_channels / 2;

            assert(weight0_shape[3] == 1);
            assert(weight1_shape[2] == 1);

            int in_group_size = in_channels[0] / groups;
            int out_group_size = out_channels / groups;
            Expr group = min(co, num_v_filters - 1) / out_group_size;

            Expr v = 0.0f;
            RDom rv(0, weight0_shape[2]);
            for (int ci = 0; ci < in_group_size; ci++) {
                v += weights0(0, rv, ci, min(co, num_v_filters - 1)) * in_1(x, y + rv - weight0_shape[2]/2, ci + group * in_group_size);
            }

            RDom rh(0, weight1_shape[3]);
            group = max(co - num_v_filters, 0) / out_group_size;
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
            func(x, y, co) = e;

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
            func(x, y, co) = sum(e);
            pointwise = false;
        } else if (name == "Downsample") {
            weights0 = decode_buffer(weight0_shape, name + "." + std::to_string(id) + ".weights", weight0_b64);
            assert(weights0.dim(0).extent() == 4);
            assert(weights0.dim(1).extent() == 4);
            assert(weights0.dim(2).extent() == in_channels[0]);
            assert(weights0.dim(3).extent() == out_channels);

            Func interleave{"interleave"};
            Var u, v;
            interleave(x, y, u, co) = in_1(2*x + u, y, co);

            Expr e = 0.0f;
            RDom r(0, 4, 0, 4);
            for (int ci = 0; ci < in_channels[0]; ci++) {
                e += weights0(r.x, r.y, ci, co) *
                    interleave(x + (r.x - 1)/2, 2*y + r.y - 1, (r.x - 1) % 2, ci);
            }
            func(x, y, co) = sum(e);

            interleave
                .compute_at(func, Var::outermost())
                .store_in(MemoryType::Stack)
                .align_storage(x, 8)
                .reorder(u, co, x, y)
                .unroll(u)
                .unroll(co)
                .vectorize(x, 16);

            may_inline = false; // Just so we can schedule the interleave somewhere good
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
            Expr e = 0.0f;
            for (int i = 0; i < values_per_group; i++) {
                e += in_1(x, y, co + i * out_channels);
            }
            func(x, y, co) = e;
            pointwise = false;
        } else if (name == "Input(Bayer)") {
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
            func(x, y, co) = in_1(x, y, co) * in_2(x, y, co);
        } else if (name == "Relu") {
            func(x, y, co) = max(0.0f, in_1(x, y, co));
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
        } else if (name == "Sub") {
            func(x, y, co) = in_1(x, y, co) - in_2(x, y, co);
        } else if (name == "Upsample") {
            // Bilinear
            Expr x_near = (x / 2) + (x % 2);
            Expr x_far = (x / 2) + 1 - (x % 2);
            Expr y_near = (y / 2) + (y % 2);
            Expr y_far = (y / 2) + 1 - (y % 2);
            func(x, y, co) =
                (9 * in_1(x_near, y_near, co) +
                 3 * (in_1(x_near, y_far, co) + in_1(x_far, y_near, co)) +
                 in_1(x_far, y_far, co)) * (1.0f / 16);
            pointwise = false;
            may_inline = false;
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
            } else {
                assert(f.eof());
                break;
            }
        }
        f.close();
    }

    void generate() {
        parse_model_file(model_file);
        for (Op &op: ops) {
            op.define(ops, raw);
        }
        output = ops.back().func;

        Var xo, yo, xi, yi;
        output
            .align_bounds(x, 16)
            .align_bounds(y, 16)
            .tile(x, y, xo, yo, x, y, 256, 128)
            .split(y, y, yi, 2)
            .reorder(co, yi, x, y, xo, yo)
            .vectorize(x, 16)
            .unroll(co)
            .unroll(yi);

        output.dim(2).set_bounds(0, 3);

        for (size_t i = 0; i < ops.size() - 1; i++)  {
            bool should_schedule = ops[i].outputs.size() > 1;
            if (ops[i].outputs.size() == 1) {
                should_schedule = !ops[ops[i].outputs[0]].pointwise;
            }
            should_schedule |= !ops[i].may_inline;
            should_schedule &= !ops[i].must_inline;
            if (should_schedule) {
                ops[i].func
                    .store_in(MemoryType::Stack)
                    .align_storage(x, 8)
                    .compute_at(output, xo)
                    .vectorize(x, 16, TailStrategy::RoundUp)
                    .split(y, y, yi, 2)
                    .reorder(yi, co, x, y)
                    .unroll(yi);

                if (ops[i].unroll_channels) {
                    ops[i].func.unroll(co);
                }

                // Not strictly necessary, but useful for debugging the definitions
                ops[i].func.bound(co, 0, ops[i].out_channels);
            } else {
                if (ops[i].unroll_channels) {
                    for (int o : ops[i].outputs) {
                        ops[o].unroll_channels = true;
                    }
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

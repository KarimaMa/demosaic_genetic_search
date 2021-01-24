HALIDE_DIR=$(dir ~/projects/Halide_true_master/distrib)

bin/model_generator: model_generator.cpp
	mkdir -p bin
	c++ -g -Wall $^ $(HALIDE_DIR)tools/GenGen.cpp -I $(HALIDE_DIR)include -L $(HALIDE_DIR)lib -lHalide -Wl,-rpath,$(HALIDE_DIR)lib -lpthread -ldl -lz -o $@

bin/halide_model.a: bin/model_generator exported.txt
	./bin/model_generator -g halide_model -o bin -e static_library,registration model_file=exported.txt target=x86-64-linux-avx2-disable_llvm_loop_opt-no_runtime
bin/halide_model.registration.cpp: bin/halide_model.a

bin/runtime.a: bin/model_generator
	./bin/model_generator -r runtime -o bin target=x86-64-linux-avx2

bin/run_model: bin/halide_model.registration.cpp  bin/halide_model.a bin/runtime.a
	c++ $(HALIDE_DIR)tools/RunGenMain.cpp -I $(HALIDE_DIR)include -I $(HALIDE_DIR)tools -lpthread -lpng -ldl -lz -ljpeg $^ -o $@

clean:
	rm -rf bin

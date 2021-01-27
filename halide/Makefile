HALIDE_DIR=$(dir /home/abadams/projects/Halide/distrib/)
RTPROCESS_DIR=$(dir /home/abadams/projects/librtprocess/install/)
BIN ?= bin
MODEL_ID ?= 0
MODEL_DIR ?= CHROMA_MODEL_SEARCH_01-20-INSERT-BINOP/models
GREEN_MODELS ?= green-01-18-pareto-files/pareto_green_asts.txt 

bin/model_generator: model_generator.cpp
	mkdir -p bin
	c++ -g -Wall $^ $(HALIDE_DIR)tools/GenGen.cpp -I $(HALIDE_DIR)include -L $(HALIDE_DIR)lib -lHalide -Wl,-rpath,$(HALIDE_DIR)lib -lpthread -ldl -lz -o $@

.PRECIOUS: bin/%/exported.txt
bin/%/exported.txt: export_ast_to_halide.py
	mkdir -p $(@D)
	python3 export_ast_to_halide.py $(GREEN_MODELS) $(MODEL_DIR)/$*/model_ast $(shell ls $(MODEL_DIR)/$*/model_*_pytorch | head -n1) $@ > bin/$*/export_log.txt

.PRECIOUS: bin/%/halide_model.a
bin/%/halide_model.a: bin/model_generator bin/%/exported.txt
	mkdir -p $(@D)
	./bin/model_generator -g halide_model -o bin/$* -e static_library,registration,stmt,assembly model_file=bin/$*/exported.txt target=x86-64-linux-avx2-disable_llvm_loop_opt-no_runtime

bin/runtime.a: bin/model_generator
	./bin/model_generator -r runtime -o bin target=x86-64-linux-avx2

bin/RunGenMain.o: $(HALIDE_DIR)tools/RunGenMain.cpp
	mkdir -p $(@D)
	c++ -O3 -I $(HALIDE_DIR)include -c $^ -o $@

.PRECIOUS: bin/%/run_model
bin/%/run_model: bin/%/halide_model.a bin/runtime.a bin/RunGenMain.o
	c++ -I $(HALIDE_DIR)include -I $(HALIDE_DIR)tools -lpthread -lpng -ldl -lz -ljpeg bin/$*/halide_model.registration.cpp $^ -o $@

bin/%/circle_out.tmp: bin/%/run_model
	$^ raw=circle.png output=$@

bin/%/circle_out.png: bin/%/circle_out.tmp
	ImageStack -load $^ -transpose c t -clamp -save $@

bin/%/bench.txt: bin/%/run_model
	$^ --benchmarks=all --benchmark_min_time=1 --parsable_output --output_extents=[4096,4096] raw=random:0:[4096,4096] >$@

# Don't forget to set OMP_NUM_THREADS=1 for running this to get a fair
# comparison to the above (or add parallelism to the Halide schedules,
# but that's just going to add noise to the entire comparison).
bin/rtprocess: reference_algorithms/demosaic.cpp
	c++ -O3 -g $^ -I $(RTPROCESS_DIR)include -I $(HALIDE_DIR)tools -L $(RTPROCESS_DIR)lib -lrtprocess -Wl,-rpath,$(RTPROCESS_DIR)lib -o $@

clean:
	rm -rf bin

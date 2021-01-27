HALIDE_DIR=$(shell echo ~/projects/Halide_master/distrib)
RTPROCESS_DIR=$(shell echo ~/projects/librtprocess/install)

# This is the combined contents of the two chroma model search folders
MODEL_DIR ?= FLAGSHIP-CHROMA_MODEL_SEARCH-INSERT-NODE12-ASTS
# The pareto green models have been dumped in the location indicated by the paths in this text file
GREEN_MODELS ?= flagship-green-pareto-files/pareto_green_asts.txt

HL_TARGET=x86-64-osx-avx2-disable_llvm_loop_opt

CXX=g++ -std=c++14

bin/ids.txt:
	mkdir -p bin
	ls $(MODEL_DIR) > $@

bin/model_generator: model_generator.cpp
	mkdir -p bin
	$(CXX) -g -Wall $^ $(HALIDE_DIR)/tools/GenGen.cpp -I $(HALIDE_DIR)/include -L $(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib -lpthread -ldl -lz -o $@

.PRECIOUS: bin/%/exported.txt
bin/%/exported.txt: export_ast_to_halide.py
	mkdir -p $(@D)
	#python3 export_ast_to_halide.py $(GREEN_MODELS) $(MODEL_DIR)/$*/model_ast $(shell ls $(MODEL_DIR)/$*/model_*_pytorch | head -n1) $@ > bin/$*/export_log.txt
	python3 export_ast_to_halide.py $(GREEN_MODELS) $(MODEL_DIR)/$*/model_ast NoWeights $@ > bin/$*/export_log.txt

.PRECIOUS: bin/%/halide_model.a
bin/%/halide_model.a: bin/model_generator bin/%/exported.txt
	mkdir -p $(@D)
	./bin/model_generator -g halide_model -o bin/$* -e static_library,registration,stmt,assembly model_file=bin/$*/exported.txt target=$(HL_TARGET)-no_runtime

bin/runtime.a: bin/model_generator
	./bin/model_generator -r runtime -o bin target=$(HL_TARGET)

bin/RunGenMain.o: $(HALIDE_DIR)/tools/RunGenMain.cpp
	mkdir -p $(@D)
	$(CXX) -O3 -I $(HALIDE_DIR)/include -c $^ -o $@

.PRECIOUS: bin/%/run_model
bin/%/run_model: bin/%/halide_model.a bin/runtime.a bin/RunGenMain.o
	$(CXX) -I $(HALIDE_DIR)/include -I $(HALIDE_DIR)/tools -lpthread -lpng -ldl -lz -ljpeg bin/$*/halide_model.registration.cpp $^ -o $@

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
	$(CXX) -O3 -g $^ -I $(RTPROCESS_DIR)/include -I $(HALIDE_DIR)/tools -L $(RTPROCESS_DIR)/lib -lrtprocess -Wl,-rpath,$(RTPROCESS_DIR)/lib -o $@

clean:
	rm -rf bin


build_all: bin/ids.txt
	cat bin/ids.txt | while read ID; do echo bin/$${ID}/run_model; done | xargs -n64 make --keep-going -j8

bench_all: bin/ids.txt 
	cat ids.txt | while read ID; do echo bin/$${ID}/bench.txt; done | xargs -n64 make --keep-going

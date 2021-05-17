#make clean

cat ids.txt | while read F; do echo bin/${F}/run_model ; done | xargs -n64 make --keep-going  -j16
cat ids.txt | while read F; do echo bin/${F}/bench.txt ; done | xargs -n64 make --keep-going 
cat ids.txt | while read F; do grep COST bin/${F}/export_log.txt | tail -n1 ; done  | cut -d: -f2 > costs.txt
cat ids.txt | while read F; do head -n1 bin/${F}/bench.txt; done | cut -d' ' -f5 > times.txt
paste ids.txt costs.txt times.txt 

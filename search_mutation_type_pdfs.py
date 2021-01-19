
full_model_pdfs = {
    "demosaicnet_search": [0.22, 0.22, 0, 0.20, 0.20, 0.16],
    "insertion_bias": {"early": [0.4, 0.5, 0, 0, 0, 0.10], "late": [0.25, 0.25, 0.1, 0.15, 0.15, 0.10]},
    "uniform": [1.0/6.0 for i in range(6)]
}

rgb8chan_pdfs = {
  "insertion_bias": {"early": [0.4, 0.6, 0, 0, 0, 0], "late": [0.3, 0.3, 0.1, 0.15, 0.15, 0]},
  "uniform": [0.2, 0.2, 0.2, 0.2, 0.2, 0]
}

green_model_pdfs = {
  "demosaicnet_search":   [0.3, 0.3, 0, 0.2, 0.2, 0],
  "insertion_bias": {"early": [0.4, 0.6, 0, 0, 0, 0], "late": [0.3, 0.3, 0.1, 0.15, 0.15, 0]},
  "uniform": [0.2, 0.2, 0.2, 0.2, 0.2, 0]
}

mutation_types_pdfs = {
  "full_model": full_model_pdfs,
  "rgb8chan": rgb8chan_pdfs,
  "green_model": green_model_pdfs  
}

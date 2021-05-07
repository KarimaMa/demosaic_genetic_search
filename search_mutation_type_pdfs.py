
full_model_pdfs = {
    "insertion_bias": {"early": [0.09, 0.09, 0.14, 0.09, 0.09, 0.14, 0.09, 0.09, 0.09, 0.09], "late": [0.1 for i in range(10)]},
    "uniform": [0.1 for i in range(10)]
}

rgb8chan_pdfs = {
  "insertion_bias": {"early": [0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1], "late": [float(1/9) for i in range(9)]},
  "uniform": [float(1/9) for i in range(9)]
}

nas_pdfs = {
  "insertion_bias": {"early": [0.18, 0.28, 0.18, 0.18, 0.18], "late": [float(1/5) for i in range(5)]},
  "uniform": [0.2, 0.2, 0.2, 0.2, 0.2]
}

green_model_pdfs = {
  "insertion_bias": {"early": [0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1], "late": [float(1/9) for i in range(9)]},
  "uniform": [float(1/9) for i in range(9)]
}

mutation_types_pdfs = {
  "full_model": full_model_pdfs,
  "nas": nas_pdfs,
  "rgb8chan": rgb8chan_pdfs,
  "green_model": green_model_pdfs  
}

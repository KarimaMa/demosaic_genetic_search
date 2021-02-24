#include <chrono>
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "halide_benchmark.h"

#include <rtprocess/librtprocess.h>

using namespace std;

#define DUMP_BAYER 0

void processFalseColorCorrection  (int W, int H, float **r, float**g, float **b, const int steps);

// C++17 has std::clamp, but we just define the same here
inline float clamp(float val, float min, float max) {
  return (val < min) ? min : (val > max) ? max : val;
}

float* load_png(const char* fname, int &x, int &y, int &channels) {
  int temp_x, temp_y, temp_channels;
  unsigned char* data = stbi_load(fname, &x, &y, &channels, 0);

  float* float_png = (float*)malloc(sizeof(float) * x * y * channels);

  // convert to float
  for (int i = 0; i < x * y * channels; i++) {
    float_png[i] = clamp(data[i] / 255.0f, 0.0f, 1.0f);
  }

  return float_png;
}


void save_png(const char* fname, float* data, int x, int y, int channels) {

  unsigned char* out_data = (unsigned char*)malloc(sizeof(unsigned char) * x * y * channels);

  // TODO: should we round?
  for (int i = 0; i < x * y * channels; i++) {
    out_data[i] = (unsigned char)(data[i] * 255.0f);
  }

  stbi_write_png(fname, x, y, channels, out_data, x * channels);

  free(out_data);

}

void create_rtprocess_structs(float* pngdata, int w, int h,
                              float ***rawdata,
                              float ***red,
                              float ***green,
                              float ***blue) {

  // rtprocess wants doubly-indirect arrays
  // and the scale should be 0.0f-65535.0f

  // create raw data struct
  *rawdata = (float**)malloc(h * sizeof(float*));
  (*rawdata)[0] = (float*)malloc(w * h * sizeof(float));
  for (int i=1; i<h; i++) {
    (*rawdata)[i] = (*rawdata)[i - 1] + w;
  }

  // mosaic the data into the raw data struct
  // bayer pattern is G r
  //                  b G
  int offset[2][2] = {{1,0},{2,1}};
  for (int i=0; i<w; i++) {
    for (int j=0; j<h; j++) {
      (*rawdata)[j][i] = pngdata[j*w + i] * 65535.f;
    }
  }

  // create output structs
  *red = (float**)malloc(h * sizeof(float*));
  (*red)[0] = (float*)malloc(w * h * sizeof(float));
  for (int i=1; i<h; i++) {
    (*red)[i] = (*red)[i - 1] + w;
  }
  *green = (float**)malloc(h * sizeof(float*));
  (*green)[0] = (float*)malloc(w * h * sizeof(float));
  for (int i=1; i<h; i++) {
    (*green)[i] = (*green)[i - 1] + w;
  }
  *blue = (float**)malloc(h * sizeof(float*));
  (*blue)[0] = (float*)malloc(w * h * sizeof(float));
  for (int i=1; i<h; i++) {
    (*blue)[i] = (*blue)[i - 1] + w;
  }
}


float* convert_rtprocess_to_float_png(int w, int h, float** red, float** green, float** blue) {
  float* ret = (float*)malloc(sizeof(float) * w * h * 3);

  for (int i=0; i<w; i++) {
    for (int j=0; j<h; j++) {
      ret[j*w*3 + i*3 + 0] = clamp(red[j][i] / 65535.f, 0.f, 1.0f);
      ret[j*w*3 + i*3 + 1] = clamp(green[j][i] / 65535.f, 0.f, 1.0f);
      ret[j*w*3 + i*3 + 2] = clamp(blue[j][i] / 65535.f, 0.f, 1.0f);
    }
  }

  return ret;
}

enum DemosaicAlgorithm { ahd, vng4, amaze, lmmse1, lmmse2, lmmse3 };

#define TIC auto _timer = chrono::steady_clock::now()
#define TOC cout << "Demosaicking time: " << chrono::duration_cast<chrono::milliseconds>( \
                                                        chrono::steady_clock::now()-_timer \
                                                        ).count() << " ms " << endl

int main(int argc, char* argv[]) {
  char* input_fname;
  char* output_fname;
  DemosaicAlgorithm alg;

  if (argc < 4 ||
      (string(argv[1]) != "ahd" &&
       string(argv[1]) != "vng4" &&
       string(argv[1]) != "lmmse1" &&
       string(argv[1]) != "lmmse2" &&
       string(argv[1]) != "lmmse3" &&
       string(argv[1]) != "amaze")) {
    std::cout << "Usage: " << argv[0] << " <ahd|vng4|amaze|lmmse1|lmmse2|lmmse3> <input> <output>\n";
    return 0;
  }

  if (string(argv[1]) == "ahd") {
    alg = ahd;
  } else if (string(argv[1]) == "vng4") {
    alg = vng4;
  } else if (string(argv[1]) == "amaze") {
    alg = amaze;
  } else if (string(argv[1]) == "lmmse1") {
    alg = lmmse1;
  } else if (string(argv[1]) == "lmmse2") {
    alg = lmmse2;
  } else if (string(argv[1]) == "lmmse3") {
    alg = lmmse3;
 }

  input_fname = argv[2];
  output_fname = argv[3];

  int x, y, n;
  float* png = load_png(input_fname, x, y, n);

  std::cout << "Loaded " << input_fname << " dimensions: " <<
    x << " x " << y << " components per pixel: " << n << "\n";

  float **rawdata;
  float **red;
  float **green;
  float **blue;

  create_rtprocess_structs(png, x, y, &rawdata, &red, &green, &blue);

  //TODO: remove callbacks from their code for better timing fidelity
  std::function<bool(double)> f([](double d) { return false; });

  // run the demosaic
  double t = 0;
  switch (alg) {
  case ahd: {
    // algorithm takes only 3 channels, and rgb correction
    std::cout << "Running AHD demosaic...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 1}};
    float rgb_cam[3][4] = { {1.0f, 1.0f, 1.0f, 1.0f},
                            {1.0f, 1.0f, 1.0f, 1.0f},
                            {1.0f, 1.0f, 1.0f, 1.0f} };
    t = Halide::Tools::benchmark([&]() {
                                     ahd_demosaic(x, y, rawdata, red, green, blue, cfarray, rgb_cam, f);
                                     processFalseColorCorrection  (x, y, red, green, blue, 1);
                                         });
    break;
  }
  case vng4: {
    // algorithm takes in two green channels
    std::cout << "Running VNG4 demosaic...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 3}};
    t = Halide::Tools::benchmark(10, 10, [&]() {
                                     vng4_demosaic(x, y, rawdata, red, green, blue, cfarray, f);
                                 });
    break;
  }
  case amaze: {
    std::cout << "Running AMAZE demosaic...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 1}};
    t = Halide::Tools::benchmark(10, 10, [&]() {
                                     amaze_demosaic(x, y, 0, 0, x, y, rawdata, red, green, blue, cfarray, f, 1.0, 0, 1.0f, 1.0f);
                                 });
    break;
  }
  case lmmse1: {
    int passcount = 1;
    std::cout << "Running LMMSE demosaic with passcount=" << passcount << "...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 1}};
    t = Halide::Tools::benchmark(10, 10, [&]() {
                                     lmmse_demosaic(x, y, rawdata, red, green, blue, cfarray, f, passcount);
                                 });
    break;
  }
  case lmmse2: {
    int passcount = 2;
    std::cout << "Running LMMSE demosaic with passcount=" << passcount << "...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 1}};
    t = Halide::Tools::benchmark(10, 10, [&]() {
                                     lmmse_demosaic(x, y, rawdata, red, green, blue, cfarray, f, passcount);
                                 });
    break;
  }
  case lmmse3: {
    int passcount = 3;
    std::cout << "Running LMMSE demosaic with passcount=" << passcount << "...\n";
    unsigned int cfarray[2][2] = {{1, 0},{2, 1}};
    t = Halide::Tools::benchmark(10, 10, [&]() {
                                     lmmse_demosaic(x, y, rawdata, red, green, blue, cfarray, f, passcount);
                                 });
    break;
  }
  }

  // convert back to png
  std::cout << "Milliseconds per megapixel: " << (t * 1e9) / (x*y) << "\n";

  float* png_output = convert_rtprocess_to_float_png(x, y, red, green, blue);

  save_png(output_fname, png_output, x, y, n);

  // dump bayer
#if DUMP_BAYER
  string bayer_fname = "/Users/kamil/temp/bayer.png";
  save_png(bayer_fname.c_str(), png, x, y, n);
#endif

  std::cout << "Wrote " << output_fname << "\n";

  return 0;
}

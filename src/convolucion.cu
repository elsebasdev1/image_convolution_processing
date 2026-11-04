// Practica 01 - Convolucion de imagenes en CPU y GPU (CUDA C)
// Compilar (desde la raiz del proyecto): nvcc -O2 src/convolucion.cu -o bin/convolucion
// Ejecutar  (desde la raiz del proyecto): ./bin/convolucion
//
// Condiciones comunes CPU/GPU: escala de grises, float, mismas mascaras, mismos datos,
// bordes con extension de ceros, arreglos de entrada y salida separados.
//
// Alcance de medicion: NO incluye lectura/escritura de archivos ni preparacion de imagenes.
// La reserva de memoria (cudaMalloc) se hace ANTES de medir y NO esta incluida.
// h2d incluye imagen + mascara. Tiempo_total_GPU = h2d + kernel + d2h.
// Cada configuracion: 1 calentamiento + REPS repeticiones medidas (mediana y desviacion estandar).
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int    REPS      = 5;      // repeticiones medidas por configuracion (minimo exigido: 5)
const double TOLERANCIA = 1e-2;  // ver justificacion en tolerancia() / main

// Verificacion de errores CUDA
#define CHECK(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { \
    std::cerr << "Error CUDA en " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err_) << std::endl; \
    exit(1); } } while (0)

// --- Implementacion CPU secuencial (referencia) ---
void convolucionCPU(const float* img_in, float* img_out, const float* mascara, int mask_size, int width, int height) {
    int offset = mask_size / 2;
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float suma = 0.0f;
            for (int mr = -offset; mr <= offset; ++mr) {
                for (int mc = -offset; mc <= offset; ++mc) {
                    int img_r = r + mr;
                    int img_c = c + mc;
                    if (img_r >= 0 && img_r < height && img_c >= 0 && img_c < width) { // fuera = cero
                        suma += img_in[img_r * width + img_c] * mascara[(mr + offset) * mask_size + (mc + offset)];
                    }
                }
            }
            img_out[r * width + c] = suma;
        }
    }
}

// --- Kernel GPU: bloques e hilos en X, un hilo por pixel de salida ---
__global__ void convolucionGPU(const float* img_in, float* img_out, const float* mascara, int mask_size, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // indice global
    if (idx >= width * height) return;                 // hilos sobrantes
    int r = idx / width;                               // fila
    int c = idx % width;                               // columna
    int offset = mask_size / 2;

    float suma = 0.0f;
    for (int mr = -offset; mr <= offset; ++mr) {
        for (int mc = -offset; mc <= offset; ++mc) {
            int img_r = r + mr;
            int img_c = c + mc;
            if (img_r >= 0 && img_r < height && img_c >= 0 && img_c < width) {
                suma += img_in[img_r * width + img_c] * mascara[(mr + offset) * mask_size + (mc + offset)];
            }
        }
    }
    img_out[idx] = suma;
}

// --- Utilidades estadisticas ---
double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double desviacion(const std::vector<double>& v) {
    double m = 0; for (double x : v) m += x; m /= v.size();
    double s = 0; for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

// Errores absolutos CPU vs GPU (sobre los float, antes de convertir a imagen)
void calcularErrores(const float* a, const float* b, int n, double& maximo, double& medio) {
    maximo = 0; medio = 0;
    for (int i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);   // NaN (pixel sin procesar) se detecta aparte
        if (!(d <= maximo)) maximo = d;
        medio += d;
    }
    medio /= n;
}

// Carga una imagen en escala de grises como float
std::vector<float> cargarImagen(const char* ruta, int& w, int& h) {
    int ch;
    unsigned char* data = stbi_load(ruta, &w, &h, &ch, 1);
    if (!data) { std::cerr << "Error: no se pudo cargar " << ruta << std::endl; return {}; }
    std::vector<float> img(data, data + (size_t)w * h);
    stbi_image_free(data);
    return img;
}

// Conversion a rango visual [0,255]: se recorta (clamp). El laplaciano suma 128 antes
// de recortar para que los valores negativos tambien se vean. Solo para guardar/visualizar.
void guardarImagen(const char* ruta, const float* datos, int w, int h, bool sumar128) {
    std::vector<unsigned char> out((size_t)w * h);
    for (size_t i = 0; i < out.size(); ++i) {
        float v = datos[i] + (sumar128 ? 128.0f : 0.0f);
        out[i] = (unsigned char)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
    }
    stbi_write_jpg(ruta, w, h, 1, out.data(), 100);
}

// Convolucion GPU simple (para las pruebas de validacion, sin mediciones)
void convolucionEnGPU(const float* h_in, float* h_out, const float* h_mask, int k, int w, int h, int tpb) {
    size_t nb = (size_t)w * h * sizeof(float);
    float *d_in, *d_out, *d_mask;
    CHECK(cudaMalloc(&d_in, nb)); CHECK(cudaMalloc(&d_out, nb)); CHECK(cudaMalloc(&d_mask, k * k * sizeof(float)));
    CHECK(cudaMemset(d_out, 0xFF, nb));   // centinela NaN: si un pixel no se procesa, queda NaN
    CHECK(cudaMemcpy(d_in, h_in, nb, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, h_mask, k * k * sizeof(float), cudaMemcpyHostToDevice));
    int blocks = (w * h + tpb - 1) / tpb;
    convolucionGPU<<<blocks, tpb>>>(d_in, d_out, d_mask, k, w, h);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, nb, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_mask);
}

// --- Prueba 1: validacion manual (laplaciano sobre matriz 3x3 con valores 1..9) ---
// Resultado calculado a mano (bordes con ceros), p. ej. centro: 2+8+4+6-4*5 = 0
bool pruebaManual(const float* mask_lap) {
    float in[9]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float esperado[9] = {2, 1, -4, -3, 0, -7, -16, -11, -22};
    float cpu[9], gpu[9];
    convolucionCPU(in, cpu, mask_lap, 3, 3, 3);
    convolucionEnGPU(in, gpu, mask_lap, 3, 3, 3, 4);   // 4 hilos/bloque -> 3 bloques, hilos sobrantes
    bool ok = true;
    for (int i = 0; i < 9; ++i) if (cpu[i] != esperado[i] || gpu[i] != esperado[i]) ok = false;
    std::cout << "[Prueba 1] Validacion manual (laplaciano 3x3): " << (ok ? "OK" : "FALLO") << "\n";
    std::cout << "  Esperado/CPU/GPU:\n";
    for (int i = 0; i < 9; ++i) std::cout << "  " << esperado[i] << " / " << cpu[i] << " / " << gpu[i] << "\n";
    return ok;
}

// --- Prueba 2: dimensiones que no son multiplo de los hilos por bloque ---
bool pruebaDimensiones(const float* mask_media_9, const float* mask_nitidez) {
    const int w = 1237, h = 911, tpb = 256;   // 1237*911 no es multiplo de 256; 1237 no es multiplo de ancho de bloque
    std::vector<float> in((size_t)w * h), cpu(in.size()), gpu(in.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((i * 31 + (i / w) * 17) % 256);
    bool ok = true;
    for (int t = 0; t < 2; ++t) {
        const float* m = (t == 0) ? mask_media_9 : mask_nitidez;
        int k = (t == 0) ? 9 : 3;
        convolucionCPU(in.data(), cpu.data(), m, k, w, h);
        convolucionEnGPU(in.data(), gpu.data(), m, k, w, h, tpb);
        int sinProcesar = 0;
        for (float v : gpu) if (std::isnan(v)) sinProcesar++;
        double emax, emed; calcularErrores(cpu.data(), gpu.data(), w * h, emax, emed);
        bool pasa = (sinProcesar == 0) && (emax <= TOLERANCIA);
        ok = ok && pasa;
        std::cout << "[Prueba 2] " << w << "x" << h << " mascara " << k << "x" << k << ", " << tpb
                  << " hilos/bloque: pixeles sin procesar=" << sinProcesar << ", err max=" << emax
                  << ", err medio=" << emed << " -> " << (pasa ? "OK" : "FALLO") << "\n";
    }
    return ok;
}

// --- Medicion de una configuracion: calentamiento + REPS repeticiones, CPU y CUDA ---
void ejecutarConfig(const char* exp, const char* filtro,
                    const float* h_mask, int k, const float* h_in, int w, int h,
                    int tpb, bool esLaplaciano, FILE* csv, FILE* resumen) {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    if (tpb > prop.maxThreadsPerBlock) {
        std::cout << "Omitido (" << tpb << " hilos/bloque no soportado por el dispositivo)\n"; return;
    }
    int n = w * h;
    size_t nb = (size_t)n * sizeof(float);
    size_t libre, total_mem; CHECK(cudaMemGetInfo(&libre, &total_mem));
    if (2 * nb + k * k * sizeof(float) > libre) { std::cout << "Omitido: memoria GPU insuficiente\n"; return; }

    int blocks = (n + tpb - 1) / tpb;
    std::cout << "\n" << exp << " | " << filtro << " " << k << "x" << k << " | " << w << "x" << h
              << " | " << tpb << " hilos/bloque, " << blocks << " bloques\n";

    std::vector<float> out_cpu(n), out_gpu(n);
    float *d_in, *d_out, *d_mask;                       // reserva previa (fuera de la medicion)
    CHECK(cudaMalloc(&d_in, nb)); CHECK(cudaMalloc(&d_out, nb)); CHECK(cudaMalloc(&d_mask, k * k * sizeof(float)));
    cudaEvent_t e[6];                                    // h2d ini/fin, kernel ini/fin, d2h ini/fin
    for (int i = 0; i < 6; ++i) CHECK(cudaEventCreate(&e[i]));

    std::vector<double> v_cpu, v_h2d, v_ker, v_d2h, v_tot;
    for (int rep = 0; rep <= REPS; ++rep) {              // rep 0 = calentamiento (no se registra)
        // GPU
        CHECK(cudaEventRecord(e[0]));
        CHECK(cudaMemcpy(d_in, h_in, nb, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_mask, h_mask, k * k * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(e[1]));
        CHECK(cudaEventRecord(e[2]));
        convolucionGPU<<<blocks, tpb>>>(d_in, d_out, d_mask, k, w, h);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(e[3]));
        CHECK(cudaEventRecord(e[4]));
        CHECK(cudaMemcpy(out_gpu.data(), d_out, nb, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(e[5]));
        CHECK(cudaEventSynchronize(e[5]));               // garantiza que todo termino
        float th2d, tker, td2h;
        CHECK(cudaEventElapsedTime(&th2d, e[0], e[1]));
        CHECK(cudaEventElapsedTime(&tker, e[2], e[3]));
        CHECK(cudaEventElapsedTime(&td2h, e[4], e[5]));
        // CPU
        auto t0 = std::chrono::steady_clock::now();
        convolucionCPU(h_in, out_cpu.data(), h_mask, k, w, h);
        auto t1 = std::chrono::steady_clock::now();
        double tcpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rep == 0) continue;
        double ttot = (double)th2d + tker + td2h;
        v_cpu.push_back(tcpu); v_h2d.push_back(th2d); v_ker.push_back(tker); v_d2h.push_back(td2h); v_tot.push_back(ttot);
        fprintf(csv, "CPU,%s,%dx%d,%d,1,1,%d,0,%.6f,0,%.6f,ms,%s\n", filtro, w, h, k, rep, tcpu, tcpu, exp);
        fprintf(csv, "CUDA,%s,%dx%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,ms,%s\n", filtro, w, h, k, tpb, blocks, rep, th2d, tker, td2h, ttot, exp);
    }
    fflush(csv);

    // Prueba 3: comparacion numerica CPU-GPU antes de convertir a imagen
    double emax, emed; calcularErrores(out_cpu.data(), out_gpu.data(), n, emax, emed);
    double mc = mediana(v_cpu), mh = mediana(v_h2d), mk = mediana(v_ker), md = mediana(v_d2h), mt = mediana(v_tot);
    double sp_k = mc / mk, sp_t = mc / mt, pct_transf = 100.0 * (mh + md) / mt;
    bool valido = emax <= TOLERANCIA;

    std::cout << "  CPU    mediana " << mc << " ms (desv " << desviacion(v_cpu) << ")\n"
              << "  H2D    mediana " << mh << " ms (desv " << desviacion(v_h2d) << ")\n"
              << "  Kernel mediana " << mk << " ms (desv " << desviacion(v_ker) << ")\n"
              << "  D2H    mediana " << md << " ms (desv " << desviacion(v_d2h) << ")\n"
              << "  Total GPU mediana " << mt << " ms (desv " << desviacion(v_tot) << ")\n"
              << "  Speedup kernel " << sp_k << "x | Speedup total " << sp_t << "x | Transferencias " << pct_transf << "%\n"
              << "  Error max " << emax << " | Error medio " << emed << " | " << (valido ? "VALIDO" : "NO VALIDO") << "\n";
    fprintf(resumen, "%s,%s,%dx%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.6g,%.6g,%s\n",
            exp, filtro, w, h, k, tpb, blocks, mc, desviacion(v_cpu), mh, mk, desviacion(v_ker), md, mt, desviacion(v_tot),
            sp_k, sp_t, pct_transf, emax, emed, valido ? "SI" : "NO");
    fflush(resumen);

    char ruta[256];
    snprintf(ruta, sizeof(ruta), "imagenes/salida/%s_%s_%dx%d_%dx%d_%dtpb.jpeg", exp, filtro, w, h, k, k, tpb);
    guardarImagen(ruta, out_gpu.data(), w, h, esLaplaciano);

    for (int i = 0; i < 6; ++i) cudaEventDestroy(e[i]);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_mask);
}

int main() {
    // Informacion del dispositivo (equipo documentado)
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    int vrt, vdrv; cudaRuntimeGetVersion(&vrt); cudaDriverGetVersion(&vdrv);
    std::cout << "GPU: " << prop.name << " | Memoria: " << prop.totalGlobalMem / (1024 * 1024) << " MB"
              << " | SMs: " << prop.multiProcessorCount << " | Compute capability: " << prop.major << "." << prop.minor
              << " | Max hilos/bloque: " << prop.maxThreadsPerBlock << "\n"
              << "CUDA runtime: " << vrt / 1000 << "." << (vrt % 100) / 10
              << " | Driver CUDA: " << vdrv / 1000 << "." << (vdrv % 100) / 10 << "\n"
              << "(CPU, RAM, SO, compilador y driver NVIDIA: documentar en el README)\n\n";

    // Mascaras
    float mask_media_3[9], mask_media_9[81], mask_media_21[441];
    for (int i = 0; i < 9; i++)   mask_media_3[i]  = 1.0f / 9.0f;
    for (int i = 0; i < 81; i++)  mask_media_9[i]  = 1.0f / 81.0f;
    for (int i = 0; i < 441; i++) mask_media_21[i] = 1.0f / 441.0f;
    float mask_nitidez[9]    = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    float mask_laplaciano[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };

    // Tolerancia: CPU y GPU hacen las mismas operaciones en float y en el mismo orden; solo pueden
    // diferir por redondeo (contraccion FMA). Cota del error ~ (k*k)*eps_float*valor_max
    // = 441 * 6e-8 * 255 ~ 6.7e-3 en el peor caso (21x21). Se usa 1e-2 (~0.004% del rango 0-255).
    std::cout << "=== VALIDACION (tolerancia = " << TOLERANCIA << ") ===\n";
    bool ok1 = pruebaManual(mask_laplaciano);
    bool ok2 = pruebaDimensiones(mask_media_9, mask_nitidez);
    if (!ok1 || !ok2) { std::cerr << "Validacion fallida, se detienen los experimentos.\n"; return 1; }

    FILE* csv = fopen("datos/mediciones.csv", "w");
    FILE* resumen = fopen("datos/resumen.csv", "w");
    if (!csv || !resumen) { std::cerr << "No se pudo crear los CSV en datos/\n"; return 1; }
    fprintf(csv, "implementation,filter,resolution,mask_size,threads_per_block,blocks,repetition,h2d_time,kernel_time,d2h_time,total_time,time_unit,experiment\n");
    fprintf(resumen, "experiment,filter,resolution,mask_size,threads_per_block,blocks,cpu_median,cpu_std,h2d_median,kernel_median,kernel_std,d2h_median,total_median,total_std,speedup_kernel,speedup_total,transfer_percent,max_abs_error,mean_abs_error,valid\n");

    int w, h;
    std::vector<float> img;

    // EXPERIMENTO 1: filtros 3x3, 4000x4000, 256 hilos
    std::cout << "\n=== EXPERIMENTO 1: FILTROS ===";
    img = cargarImagen("imagenes/entrada/prueba_4000.jpeg", w, h);
    if (!img.empty()) {
        ejecutarConfig("exp1", "media",      mask_media_3,    3, img.data(), w, h, 256, false, csv, resumen);
        ejecutarConfig("exp1", "nitidez",    mask_nitidez,    3, img.data(), w, h, 256, false, csv, resumen);
        ejecutarConfig("exp1", "laplaciano", mask_laplaciano, 3, img.data(), w, h, 256, true,  csv, resumen);
    }

    // EXPERIMENTO 2: resolucion, media 9x9, 256 hilos
    std::cout << "\n=== EXPERIMENTO 2: RESOLUCION ===";
    const char* archivos[] = {"imagenes/entrada/prueba_2000.jpeg", "imagenes/entrada/prueba_4000.jpeg", "imagenes/entrada/prueba_6000.jpeg"};
    for (int i = 0; i < 3; i++) {
        img = cargarImagen(archivos[i], w, h);
        if (!img.empty()) ejecutarConfig("exp2", "media", mask_media_9, 9, img.data(), w, h, 256, false, csv, resumen);
    }

    // EXPERIMENTO 3 y 4 usan 4000x4000
    img = cargarImagen("imagenes/entrada/prueba_4000.jpeg", w, h);
    if (!img.empty()) {
        // EXPERIMENTO 3: tamano de mascara, 256 hilos
        std::cout << "\n=== EXPERIMENTO 3: TAMANO DE MASCARA ===";
        ejecutarConfig("exp3", "media", mask_media_3,  3,  img.data(), w, h, 256, false, csv, resumen);
        ejecutarConfig("exp3", "media", mask_media_9,  9,  img.data(), w, h, 256, false, csv, resumen);
        ejecutarConfig("exp3", "media", mask_media_21, 21, img.data(), w, h, 256, false, csv, resumen);

        // EXPERIMENTO 4: hilos por bloque, media 9x9
        std::cout << "\n=== EXPERIMENTO 4: HILOS POR BLOQUE ===";
        int configs[] = {128, 256, 512};
        for (int t : configs)
            ejecutarConfig("exp4", "media", mask_media_9, 9, img.data(), w, h, t, false, csv, resumen);
    }

    fclose(csv); fclose(resumen);
    std::cout << "\nPruebas finalizadas. CSV en datos/mediciones.csv y datos/resumen.csv\n";
    return 0;
}

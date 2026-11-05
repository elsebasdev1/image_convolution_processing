[sebas@archlinux practica1]$ ./bin/convolucion.bin
GPU: NVIDIA GeForce RTX 3060 Laptop GPU | Memoria: 5857 MB | SMs: 30 | CC: 8.6 | Max hilos/bloque: 1024
CUDA runtime: 13.4 | Driver CUDA: 13.4

=== VALIDACION (tolerancia = 0.01) ===
[Prueba 1] Validacion manual (laplaciano 3x3): OK
  Esperado/CPU/GPU: 2/2/2, 1/1/1, -4/-4/-4, -3/-3/-3, 0/0/0, -7/-7/-7, -16/-16/-16, -11/-11/-11, -22/-22/-22
[Prueba 2] 1237x911 mascara 9x9, 256 hilos/bloque: sin procesar=0, err max=1.52588e-05, err medio=3.82612e-06 -> OK
[Prueba 2] 1237x911 mascara 3x3, 256 hilos/bloque: sin procesar=0, err max=0, err medio=0 -> OK

=== EXPERIMENTO 1: FILTROS ===
exp1 | media 3x3 | 4000x4000 | 256 hilos/bloque, 62500 bloques
  CPU: 97.57 ms | Kernel: 0.704 ms | Total GPU: 11.39 ms | Speedup Ker: 138.54x | Speedup Tot: 8.57x | Transf: 93.81%
  Error max: 4.57764e-05 | Error medio: 1.93399e-06 | VALIDO
exp1 | nitidez 3x3 | 4000x4000 | 256 hilos/bloque, 62500 bloques
  CPU: 98.74 ms | Kernel: 0.707 ms | Total GPU: 11.47 ms | Speedup Ker: 139.75x | Speedup Tot: 8.61x | Transf: 94.22%
  Error max: 0 | Error medio: 0 | VALIDO
exp1 | laplaciano 3x3 | 4000x4000 | 256 hilos/bloque, 62500 bloques
  CPU: 97.75 ms | Kernel: 0.704 ms | Total GPU: 11.33 ms | Speedup Ker: 138.79x | Speedup Tot: 8.63x | Transf: 93.79%
  Error max: 0 | Error medio: 0 | VALIDO

=== EXPERIMENTO 2: RESOLUCION ===
exp2 | media 9x9 | 2000x2000 | 256 hilos/bloque: CPU 194.30 ms | Kernel 0.995 ms | Total 3.89 ms | Speedup Ker: 195.25x
exp2 | media 9x9 | 4000x4000 | 256 hilos/bloque: CPU 779.59 ms | Kernel 3.928 ms | Total 16.11 ms | Speedup Ker: 198.47x
exp2 | media 9x9 | 6000x6000 | 256 hilos/bloque: CPU 1759.29 ms | Kernel 8.800 ms | Total 32.46 ms | Speedup Ker: 199.91x

=== EXPERIMENTO 3: TAMANO DE MASCARA ===
exp3 | media 3x3 | 4000x4000: CPU 98.93 ms | Kernel 0.702 ms | Total GPU 12.99 ms | Speedup Tot: 7.62x (Transf 95.28%)
exp3 | media 9x9 | 4000x4000: CPU 784.55 ms | Kernel 3.924 ms | Total GPU 16.22 ms | Speedup Tot: 48.36x (Transf 75.82%)
exp3 | media 21x21 | 4000x4000: CPU 4453.99 ms | Kernel 20.80 ms | Total GPU 32.88 ms | Speedup Tot: 135.46x (Transf 36.66%)

=== EXPERIMENTO 4: HILOS POR BLOQUE ===
exp4 | 128 hilos/bloque: Kernel 3.913 ms (± 0.004) | Total GPU 16.09 ms | Speedup Ker: 200.50x
exp4 | 256 hilos/bloque: Kernel 3.922 ms (± 0.000) | Total GPU 16.09 ms | Speedup Ker: 200.19x
exp4 | 512 hilos/bloque: Kernel 3.974 ms (± 0.000) | Total GPU 16.20 ms | Speedup Ker: 197.83x


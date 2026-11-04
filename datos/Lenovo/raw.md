=== EXPERIMENTO 1: FILTROS (Imagen 4000x4000) ===

--------------------------------------------------
Ejecutando: Media_3x3 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 740.297 ms
Transferencia H2D    : 9.92051 ms
Ejecucion Kernel     : 2.83674 ms
Transferencia D2H    : 12.7527 ms
Tiempo Total GPU     : 25.6516 ms
Aceleracion (Kernel) : 260.968x
Error Abs. Maximo    : 4.57764e-05

--------------------------------------------------
Ejecutando: Nitidez_3x3 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 702.528 ms
Transferencia H2D    : 9.99856 ms
Ejecucion Kernel     : 2.52941 ms
Transferencia D2H    : 11.1191 ms
Tiempo Total GPU     : 23.6732 ms
Aceleracion (Kernel) : 277.744x
Error Abs. Maximo    : 0

--------------------------------------------------
Ejecutando: Laplaciano_3x3 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 699.612 ms
Transferencia H2D    : 10.2037 ms
Ejecucion Kernel     : 2.52845 ms
Transferencia D2H    : 11.4683 ms
Tiempo Total GPU     : 24.2725 ms
Aceleracion (Kernel) : 276.696x
Error Abs. Maximo    : 0


=== EXPERIMENTO 2: RESOLUCION (Filtro Media 9x9) ===

--------------------------------------------------
Ejecutando: Res_2000x2000 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 1475.41 ms
Transferencia H2D    : 10.4925 ms
Ejecucion Kernel     : 10.6702 ms
Transferencia D2H    : 9.85274 ms
Tiempo Total GPU     : 31.0674 ms
Aceleracion (Kernel) : 138.274x
Error Abs. Maximo    : 0.000167847

--------------------------------------------------
Ejecutando: Res_4000x4000 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 5935.46 ms
Transferencia H2D    : 39.4388 ms
Ejecucion Kernel     : 89.1499 ms
Transferencia D2H    : 39.1271 ms
Tiempo Total GPU     : 167.799 ms
Aceleracion (Kernel) : 66.5784x
Error Abs. Maximo    : 0.000167847

--------------------------------------------------
Ejecutando: Res_6000x6000 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 13275 ms
Transferencia H2D    : 86.994 ms
Ejecucion Kernel     : 184.103 ms
Transferencia D2H    : 25.5311 ms
Tiempo Total GPU     : 296.692 ms
Aceleracion (Kernel) : 72.1064x
Error Abs. Maximo    : 0.000167847


=== EXPERIMENTO 3: HILOS POR BLOQUE (Imagen 4000x4000, Media 9x9) ===

--------------------------------------------------
Ejecutando: Hilos_128 (128 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 5887.17 ms
Transferencia H2D    : 39.1082 ms
Ejecucion Kernel     : 90.3441 ms
Transferencia D2H    : 39.1117 ms
Tiempo Total GPU     : 168.649 ms
Aceleracion (Kernel) : 65.1638x
Error Abs. Maximo    : 0.000167847

--------------------------------------------------
Ejecutando: Hilos_256 (256 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 5894.54 ms
Transferencia H2D    : 39.2946 ms
Ejecucion Kernel     : 89.1484 ms
Transferencia D2H    : 16.6354 ms
Tiempo Total GPU     : 145.161 ms
Aceleracion (Kernel) : 66.1205x
Error Abs. Maximo    : 0.000167847

--------------------------------------------------
Ejecutando: Hilos_512 (512 hilos/bloque)
--------------------------------------------------
Tiempo CPU           : 5882.98 ms
Transferencia H2D    : 39.5136 ms
Ejecucion Kernel     : 90.5895 ms
Transferencia D2H    : 39.14 ms
Tiempo Total GPU     : 169.332 ms
Aceleracion (Kernel) : 64.941x
Error Abs. Maximo    : 0.000167847

Pruebas finalizadas. Revisa los archivos de salida generados.
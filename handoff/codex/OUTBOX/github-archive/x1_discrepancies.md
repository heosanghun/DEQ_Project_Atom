# TASK-C002 フラグ差分 / 플래그 불일치

작성자: Codex | 2026-10-20T00:46:02.619591+09:00

로컬 값은 Windows multiprocessing 오류가 있는 첫 시도의 진단용 산출물입니다. 수식 동등성 재채점의 확정 결과가 아닙니다.

모든 원문은 x1_recheck_results.json differences 배열에 보존되어 있습니다.

| label | id | flag | CC | local | prediction boxed | prediction lenient | gold |
|---|---:|---|---:|---:|---|---|---|
| C_fmt | 445 | strict | 1 | 0 | 16/3 | 16/3 | \frac{16}{3} |
| C_fmt | 445 | lenient | 1 | 0 | 16/3 | 16/3 | \frac{16}{3} |
| C_fmt | 566 | lenient | 1 | 0 | None | \frac{2009(2009-1)}{2009} | 2008 |
| C_fmt | 906 | strict | 1 | 0 | \frac{175}{28} | \frac{175}{28} | \frac{25}{4} |
| C_fmt | 906 | lenient | 1 | 0 | \frac{175}{28} | \frac{175}{28} | \frac{25}{4} |
| C_fmt | 1062 | lenient | 1 | 0 | None | \frac{20}{4} | 5 |
| C_fmt | 1387 | lenient | 1 | 0 | None | \frac{-1+5}{2} | 2 |
| C_fmt | 1503 | lenient | 1 | 0 | None | \frac{0.64P}{P} | 64\% |
| C_fmt | 1875 | lenient | 1 | 0 | None | \frac{25 \cdot 24 \cdot 23 \cdot 22 \cdot 21}{5 \cdot 4 \cdot 3 \cdot 2 \cdot 1} | 53130 |
| C_fmt | 2280 | lenient | 1 | 0 | None | \frac{50 \cdot 49}{2} | 1225 |
| C_fmt | 2845 | lenient | 1 | 0 | None | \frac{27}{30} | \frac{9}{10} |
| C_fmt | 3831 | strict | 1 | 0 | P | P | \text{(P)} |
| C_fmt | 3831 | lenient | 1 | 0 | P | P | \text{(P)} |
| C_fmt | 4032 | strict | 1 | 0 | -\frac{15}{4} | -\frac{15}{4} | -\frac{15}4 |
| C_fmt | 4032 | lenient | 1 | 0 | -\frac{15}{4} | -\frac{15}{4} | -\frac{15}4 |
| X1O47 | 699 | strict | 1 | 0 | \dfrac{36}{6} | \dfrac{36}{6} | 6 |
| X1O47 | 699 | lenient | 1 | 0 | \dfrac{36}{6} | \dfrac{36}{6} | 6 |
| X1O47 | 1224 | strict | 1 | 0 | \$ 144 | \$ 144 | 144 |
| X1O47 | 1224 | lenient | 1 | 0 | \$ 144 | \$ 144 | 144 |
| X1O47 | 1410 | strict | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| X1O47 | 1410 | lenient | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| X1O47 | 2369 | strict | 1 | 0 | 2/8 | 2/8 | \frac{1}{4} |
| X1O47 | 2369 | lenient | 1 | 0 | 2/8 | 2/8 | \frac{1}{4} |
| X1O47 | 3873 | lenient | 1 | 0 | None | \frac{10}{10} | 1 |
| shuf_X1O47 | 699 | strict | 1 | 0 | \dfrac{36}{6} | \dfrac{36}{6} | 6 |
| shuf_X1O47 | 699 | lenient | 1 | 0 | \dfrac{36}{6} | \dfrac{36}{6} | 6 |
| shuf_X1O47 | 1224 | strict | 1 | 0 | \$ 144 | \$ 144 | 144 |
| shuf_X1O47 | 1224 | lenient | 1 | 0 | \$ 144 | \$ 144 | 144 |
| shuf_X1O47 | 1410 | strict | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| shuf_X1O47 | 1410 | lenient | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| shuf_X1O47 | 2358 | lenient | 1 | 0 | None | there are $16$ possible outcomes for the first game | 16 |
| X1O49 | 274 | strict | 1 | 0 | \mathbb{R}\setminus\{-27\} | \mathbb{R}\setminus\{-27\} | (-\infty,-27)\cup(-27,\infty) |
| X1O49 | 274 | lenient | 1 | 0 | \mathbb{R}\setminus\{-27\} | \mathbb{R}\setminus\{-27\} | (-\infty,-27)\cup(-27,\infty) |
| X1O49 | 379 | strict | 1 | 0 | 5-6 = -1 | 5-6 = -1 | -1 |
| X1O49 | 379 | lenient | 1 | 0 | 5-6 = -1 | 5-6 = -1 | -1 |
| X1O49 | 1156 | strict | 1 | 0 | 2(xy + 5x + 10y + 50) | 2(xy + 5x + 10y + 50) | 2xy + 10x + 20y + 100 |
| X1O49 | 1156 | lenient | 1 | 0 | 2(xy + 5x + 10y + 50) | 2(xy + 5x + 10y + 50) | 2xy + 10x + 20y + 100 |
| X1O49 | 1224 | strict | 1 | 0 | \$ 144 | \$ 144 | 144 |
| X1O49 | 1224 | lenient | 1 | 0 | \$ 144 | \$ 144 | 144 |
| X1O49 | 1343 | strict | 1 | 0 | -5.5 | -5.5 | -\frac{11}{2} |
| X1O49 | 1343 | lenient | 1 | 0 | -5.5 | -5.5 | -\frac{11}{2} |
| X1O49 | 2212 | strict | 1 | 0 | 1/64 | 1/64 | \frac{1}{64} |
| X1O49 | 2212 | lenient | 1 | 0 | 1/64 | 1/64 | \frac{1}{64} |
| X1O49 | 2222 | strict | 1 | 0 | \frac{16}{32}=\frac{2}{4}=\frac{1}{2} | \frac{16}{32}=\frac{2}{4}=\frac{1}{2} | \frac{1}{2} |
| X1O49 | 2222 | lenient | 1 | 0 | \frac{16}{32}=\frac{2}{4}=\frac{1}{2} | \frac{16}{32}=\frac{2}{4}=\frac{1}{2} | \frac{1}{2} |
| X1O49 | 3831 | strict | 1 | 0 | \text{P} | \text{P} | \text{(P)} |
| X1O49 | 3831 | lenient | 1 | 0 | \text{P} | \text{P} | \text{(P)} |
| shuf_X1O49 | 379 | strict | 1 | 0 | 5-2\cdot 3 = -1 | 5-2\cdot 3 = -1 | -1 |
| shuf_X1O49 | 379 | lenient | 1 | 0 | 5-2\cdot 3 = -1 | 5-2\cdot 3 = -1 | -1 |
| shuf_X1O49 | 983 | strict | 1 | 0 | \frac{3}{1} = 3 | \frac{3}{1} = 3 | 3 |
| shuf_X1O49 | 983 | lenient | 1 | 0 | \frac{3}{1} = 3 | \frac{3}{1} = 3 | 3 |
| shuf_X1O49 | 1224 | strict | 1 | 0 | \$144 | \$144 | 144 |
| shuf_X1O49 | 1224 | lenient | 1 | 0 | \$144 | \$144 | 144 |
| shuf_X1O49 | 1410 | strict | 1 | 0 | \frac 45 | \frac 45 | \frac{4}{5} |
| shuf_X1O49 | 1410 | lenient | 1 | 0 | \frac 45 | \frac 45 | \frac{4}{5} |
| shuf_X1O49 | 2769 | strict | 1 | 0 | \frac{25}{5} \pi = 10 \pi | \frac{25}{5} \pi = 10 \pi | 10\pi |
| shuf_X1O49 | 2769 | lenient | 1 | 0 | \frac{25}{5} \pi = 10 \pi | \frac{25}{5} \pi = 10 \pi | 10\pi |
| shuf_X1O49 | 3278 | strict | 1 | 0 | 20 | 20 | 20\text{ square units} |
| shuf_X1O49 | 3278 | lenient | 1 | 0 | 20 | 20 | 20\text{ square units} |
| shuf_X1O49 | 3530 | strict | 1 | 0 | (x+2)^2 - 81x^4 | (x+2)^2 - 81x^4 | (-9x^2+x+2)(9x^2+x+2) |
| shuf_X1O49 | 3530 | lenient | 1 | 0 | (x+2)^2 - 81x^4 | (x+2)^2 - 81x^4 | (-9x^2+x+2)(9x^2+x+2) |
| X1O50 | 645 | strict | 1 | 0 | \dfrac{4}{9} | \dfrac{4}{9} | \frac{4}{9} |
| X1O50 | 645 | lenient | 1 | 0 | \dfrac{4}{9} | \dfrac{4}{9} | \frac{4}{9} |
| X1O50 | 1056 | strict | 1 | 0 | 4a^3 + 2a^2 - a^2 | 4a^3 + 2a^2 - a^2 | 4a^3 + a^2 |
| X1O50 | 1056 | lenient | 1 | 0 | 4a^3 + 2a^2 - a^2 | 4a^3 + 2a^2 - a^2 | 4a^3 + a^2 |
| X1O50 | 1464 | strict | 1 | 0 | \$40$. | \$40$. | 40\% |
| X1O50 | 1464 | lenient | 1 | 0 | \$40$. | \$40$. | 40\% |
| X1O50 | 3530 | strict | 1 | 0 | (-81x^4+x^2+4x+4) | (-81x^4+x^2+4x+4) | (-9x^2+x+2)(9x^2+x+2) |
| X1O50 | 3530 | lenient | 1 | 0 | (-81x^4+x^2+4x+4) | (-81x^4+x^2+4x+4) | (-9x^2+x+2)(9x^2+x+2) |
| X1O50 | 3533 | strict | 1 | 0 | \pm1,\pm\sqrt{3} | \pm1,\pm\sqrt{3} | -\sqrt{3},-1,1,\sqrt{3} |
| X1O50 | 3533 | lenient | 1 | 0 | \pm1,\pm\sqrt{3} | \pm1,\pm\sqrt{3} | -\sqrt{3},-1,1,\sqrt{3} |
| shuf_X1O50 | 699 | strict | 1 | 0 | \frac{36}{6} | \frac{36}{6} | 6 |
| shuf_X1O50 | 699 | lenient | 1 | 0 | \frac{36}{6} | \frac{36}{6} | 6 |
| shuf_X1O50 | 1130 | strict | 1 | 0 | 1089-689=400 | 1089-689=400 | 400 |
| shuf_X1O50 | 1130 | lenient | 1 | 0 | 1089-689=400 | 1089-689=400 | 400 |
| shuf_X1O50 | 1410 | strict | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| shuf_X1O50 | 1410 | lenient | 1 | 0 | \frac45 | \frac45 | \frac{4}{5} |
| shuf_X1O50 | 2280 | strict | 1 | 0 | 50\cdot 49/2 | 50\cdot 49/2 | 1225 |
| shuf_X1O50 | 2280 | lenient | 1 | 0 | 50\cdot 49/2 | 50\cdot 49/2 | 1225 |

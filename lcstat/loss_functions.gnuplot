set term png size 700, 500
set xrange [-10:10]

set xtics 0.2
set ytics 0.2

set for [i=1:8] linetype i dashtype i lw 4

set grid
set xzeroaxis lw 2
set yzeroaxis lw 2

plot [-0.4:2][-0.2:1.7] \
    (x < 1 ? 1 - x : 0) title 'V(x) = 1 - x' lw 2, \
    (x < 1 ? (1 - x)**2 : 0) title 'Q(x) = (1 - x)^2' lw 2, \
    (x < 1 ? (1 - x)**(1.5) : 0) title 'Q^3_2(x) = (1 - x)^(3/2)' lw 2, \
    (x < 1 ? (1 - x)**3 : 0) title 'Q3(x) = (1 - x)^3' lw 2, \
    (x < 1 ? (1 - x)**4 : 0) title 'Q4(x) = (1 - x)^4' lw 2, \
    (log(1+exp(-x))/log(2)) title 'L(x) = log2(1 + e^-m)' lw 2, \
    (2/(1+exp(x))) title 'S(m) = 2 * (1  + e^m)^-1' lw 2, \
    (exp(-x)) title 'E(x) = exp(-x)' lw 2

# set key left box

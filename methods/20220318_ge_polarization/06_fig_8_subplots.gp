set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator '\t'
set size 0.3,0.3
set style data histogram
set style histogram cluster gap 1
set style fill solid noborder
set xrange [-1:1]
set ytics 20 

set xlabel "NOMINATE"
set ylabel "# Congressmen"
set palette defined (-1 "#4393C3", 0 "#bdbdbd", 1 "#D6604D")
unset colorbox

set output '81_nodes_hist.eps'
plot '81_nodes_hist.csv' u 1:2:1 with boxes palette notitle
set output '88_nodes_hist.eps'
plot '88_nodes_hist.csv' u 1:2:1 with boxes palette notitle
set output '95_nodes_hist.eps'
plot '95_nodes_hist.csv' u 1:2:1 with boxes palette notitle
set output '102_nodes_hist.eps'
plot '102_nodes_hist.csv' u 1:2:1 with boxes palette notitle
set output '109_nodes_hist.eps'
plot '109_nodes_hist.csv' u 1:2:1 with boxes palette notitle
set output '116_nodes_hist.eps'
plot '116_nodes_hist.csv' u 1:2:1 with boxes palette notitle

reset session

set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator '\t'
set size 1.25,0.25

set ylabel "{/Symbol d}_{G,o}"
set xlabel "Congress"
set xrange [80:117]
set xtics 2
set ytics 2

set output 'congress_pol.eps'
plot 'congress_pol.csv' u 1:2 with lines lc rgb "#e41a1c" lw 4 notitle

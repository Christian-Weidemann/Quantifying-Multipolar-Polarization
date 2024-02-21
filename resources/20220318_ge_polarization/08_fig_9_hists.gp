set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator '\t'
set size 0.45,0.45
set ytics 1
set style data histogram
set style histogram cluster gap 1
set style fill solid noborder
set xrange [-1.02:1.02]
set yrange [0:10]
#set format y "%.1l*10^{%L}"
unset colorbox

set xlabel "o"
set ylabel "# Nodes"
set palette defined (-1 '#4393C3', 0 '#cccccc', 1 '#D6604D')

set output "grid_graph_ts_hists0.eps"
plot "grid_graph_ts_hists" u 1:2:1 with boxes palette notitle

set output "grid_graph_ts_hists1.eps"
plot "grid_graph_ts_hists" u 1:3:1 with boxes palette notitle

set output "grid_graph_ts_hists4.eps"
plot "grid_graph_ts_hists" u 1:6:1 with boxes palette notitle

set output "grid_graph_ts_hists9.eps"
plot "grid_graph_ts_hists" u 1:11:1 with boxes palette notitle


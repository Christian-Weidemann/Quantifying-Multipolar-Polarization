set terminal postscript eps enhanced color solid rounded linewidth 1
set size 0.45,0.45
set style fill solid
set ylabel "AVG Neighbor Opinion"
set xlabel "Opinion"
set lmargin at screen 0.1
set bmargin at screen 0.1
set tmargin at screen 0.42

set palette defined (0 '#fff5f0', 0.5 '#fb6a4a', 1 '#67000d')
set xtics 0.5
set ytics 0.5
set xrange [-1:1]
set yrange [-1:1]

unset colorbox
set view map
set dgrid3d 100,100

list = system('ls *kde*')

do for [f in list] {
   set output sprintf('%s.eps', substr(f, 1, strlen(f) - 4))
   splot f u 1:2:3 with pm3d notitle
}

reset session

set terminal postscript eps enhanced color solid rounded linewidth 1
set size 0.45,0.45
set style fill solid
set ylabel "Influenced Set Opinion"
set xlabel "Seed Opinion"

set palette defined (0 '#fcbba1', 0.5 '#ef3b2c', 1 '#67000d')
set xtics 0.5
set ytics 0.5
set xrange [-1:1]
set yrange [-1:1]
unset colorbox
set bars 2

list = system('ls sir*')

do for [f in list] {
   set output sprintf('%s.eps', substr(f, 1, strlen(f) - 4))
   plot f u 1:2:($2-$3):($2+$3):2:4 w candlesticks lw 2 palette notitle whiskerbars
}


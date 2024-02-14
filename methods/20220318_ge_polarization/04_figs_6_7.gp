set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator '\t'
set size 0.45,0.45
set style data histogram
set style histogram cluster gap 1
set style fill solid noborder
set xrange [-1:1]

set xlabel "Polarity"
set ylabel "# Users"
set palette defined (-1 "#4393C3", 0 "#bdbdbd", 1 "#D6604D")
unset colorbox

set output 'abortion_user_scores_hist.eps'
plot 'abortion_user_scores_hist.csv' u 1:2:1 with boxes palette notitle

set output 'election_user_scores_hist.eps'
plot 'election_user_scores_hist.csv' u 1:2:1 with boxes palette notitle

set output 'guncontrol_user_scores_hist.eps'
plot 'guncontrol_user_scores_hist.csv' u 1:2:1 with boxes palette notitle

set output 'obama_user_scores_hist.eps'
plot 'obama_user_scores_hist.csv' u 1:2:1 with boxes palette notitle

set output 'second_debate_user_scores_hist.eps'
plot 'second_debate_user_scores_hist.csv' u 1:2:1 with boxes palette notitle

set output 'vp_debate_user_scores_hist.eps'
plot 'vp_debate_user_scores_hist.csv' u 1:2:1 with boxes palette notitle
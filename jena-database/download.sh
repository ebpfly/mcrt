#!/bin/bash
BASE="https://raw.githubusercontent.com/cdominik/optool/refs/heads/master/lnk_data"
FILES="pyr-mg100-Dorschner1995 pyr-mg95-Dorschner1995 pyr-mg80-Dorschner1995 pyr-mg60-Dorschner1995 pyr-mg50-Dorschner1995 pyr-mg40-Dorschner1995 ol-mg50-Dorschner1995 ol-mg40-Dorschner1995 ol-c-mg100-Suto2006 ol-c-mg95-Fabian2001 ol-c-mg00-Fabian2001 pyr-c-mg96-Jaeger1998 astrosil-Draine2003 c-z-Zubko1996 c-p-Preibisch1993 c-gra-Draine2003 c-nano-Mutschke2004 c-org-Henning1996 sio2-Kitamura2007 cor-c-Koike1995 fe-c-Henning1996 fes-Henning1996 sic-Draine1993 h2o-w-Warren2008 h2o-a-Hudgins1993 co2-w-Warren1986 co2-a-Gerakines2020 co2-c-Gerakines2020 co-a-Palumbo2006 nh3-m-Martonchik1983 ch4-a-Gerakines2020 ch4-c-Gerakines2020 ch3oh-a-Gerakines2020 ch3oh-c-Gerakines2020"
for f in $FILES; do
  echo "Downloading ${f}.lnk..."
  curl -s "${BASE}/${f}.lnk" -o "${f}.lnk"
done
echo "Done. Downloaded $(ls *.lnk | wc -l) files."

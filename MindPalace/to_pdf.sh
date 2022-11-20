#!/bin/bash

echo $1

fname=$(basename "$1")

echo $fname

cat "$1" | sd -s "\$\$\begin{align}" "\begin{align}" | sd -s "\end{align}\$\$" "\end{align}" > "/tmp/$fname"

pandoc -o output.pdf --include-in-header ./preamble.sty "/tmp/$fname"

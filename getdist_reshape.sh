#!/bin/bash

cd ~/Dropbox/LUPM/Lya/MCMC_lya/

CN='2020-06-27_Tpowerlaw_lyaprior_mike-hires'

sort chain_${CN}.dat | uniq -c > ~/Desktop/LYA_chains/chain_${CN}.txt

rm chain_${CN}.dat

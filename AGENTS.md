# AGENTS

## Background

This code is written in Ocaml and supports a package which goal is to provide a semantic type layor over tensors.

That is, we are using the `raven` package and supporting the `rune` for tensors as well as maybe `Nx` since they are  intertwined. 

The exhuastive roadmap which contains a lot more detail is in @ROADMAP.md.



## Architecture

For all intents and purposes, this is a zero-runtime package. Of course, Ocaml uses `gc` which technically isn't fully compiletime, but this is quasi-zero-runtime. The goal is to validate *without* running to avoid any expensive computations taking place.



### Semantic Boundary

This is where the answers, informatoin, and ideas will go from Phase 0.
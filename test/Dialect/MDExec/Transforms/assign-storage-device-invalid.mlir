// RUN: not mdir-opt %s --md-exec-assign-storage="memory=disk" 2>&1 \
// RUN: | FileCheck %s --check-prefix=MEMORY
// MEMORY: expected the memory 'host' or 'device', got 'disk'

md.particle_set @atoms

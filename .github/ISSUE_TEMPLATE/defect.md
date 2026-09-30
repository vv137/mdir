---
name: Defect
about: A run that fails, crashes, or gives results that look wrong
labels: defect
---

## What happened

<!-- The command, what you expected, and what happened instead. Paste the
     message of the error, or the lines of the log that look wrong. -->

## Report

<!-- Run the following and attach report.tar.gz. It holds the version, the
     devices, the inputs (with hashes; those over 16 MiB are not copied),
     and the program at each stage. See docs/debugging.md.

       mdir bug-report run.toml -o report --run
       tar czf report.tar.gz report

     If the run takes long, leave out --run and paste the log instead. -->

## Anything else

<!-- Whether it happens on the CPU and the GPU, with double and mixed
     precision, with MDRT_WAIT=1, or under compute-sanitizer. -->

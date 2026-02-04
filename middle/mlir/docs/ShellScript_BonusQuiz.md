# Shell Script Bonus Quiz

## Conceptual Questions

### Q1: Why `set -e`?
What happens WITHOUT `set -e` if a command fails?

A) Script stops immediately
B) Script continues to next command
C) Script prints error and stops
D) Script restarts

### Q2: Variable Assignment
Which is correct?

A) `VAR = "hello"`
B) `VAR ="hello"`
C) `VAR= "hello"`
D) `VAR="hello"`

### Q3: Quoting
What does `echo $HOME` vs `echo "$HOME"` difference matter for?

A) No difference ever
B) When path contains spaces
C) Only on Linux
D) Only for numbers

### Q4: Command Substitution
What does `$(dirname "$0")` return if script is `/home/user/scripts/test.sh`?

A) `test.sh`
B) `/home/user/scripts`
C) `/home/user`
D) `scripts`

### Q5: Pipe
What does `|` do in `echo "hello" | wc -c`?

A) Runs both commands in parallel
B) Sends output of left command as input to right command
C) Compares the two commands
D) Runs right command only if left succeeds

### Q6: Exit Codes
What does `exit 0` mean?

A) Error occurred
B) Success
C) Exit after 0 seconds
D) Return 0 results

### Q7: Test Conditions
What does `[ -z "$VAR" ]` test?

A) VAR equals "z"
B) VAR is a file
C) VAR is empty/zero-length
D) VAR is a number

---

## Code Writing

### Q8: Write a script that:
- Takes a filename as argument
- Checks if file exists
- If exists: prints "Found: <filename>"
- If not: prints "Not found" and exits with error

```bash
#!/bin/bash
set -e

FILE="???"

if [ ??? "$FILE" ]; then
  echo "Found: $FILE"
else
  echo "Not found"
  exit ???
fi
```

### Q9: Write a pipeline that:
- Reads a file
- Converts to uppercase
- Saves to new file

```bash
#!/bin/bash
INPUT="$1"
OUTPUT="$2"

??? "$INPUT" | ??? '[:lower:]' '[:upper:]' ??? "$OUTPUT"
```

---

## Answers

```
Conceptual:
Q1: B - Script continues (that's why set -e is important!)
Q2: D - No spaces around = in bash
Q3: B - Quoted variables handle spaces correctly
Q4: B - /home/user/scripts (the directory part)
Q5: B - Pipe sends stdout to stdin
Q6: B - Exit code 0 = success
Q7: C - -z tests if string is empty

Code:
Q8: $1, -f, 1
Q9: cat, tr, > (or use: cat "$INPUT" | tr '[:lower:]' '[:upper:]' > "$OUTPUT")
```

---

## Extra Practice

Try modifying `full_pipeline.sh` to:

1. **Add a `--help` option**
```bash
if [ "$1" = "--help" ]; then
  echo "Usage: $0 <input.mlir>"
  echo "Runs full Gawee compilation pipeline"
  exit 0
fi
```

2. **Save output to file**
```bash
OUTPUT="${INPUT%.mlir}_lowered.mlir"  # Replace .mlir with _lowered.mlir
echo "$STEP3" > "$OUTPUT"
echo "Saved to: $OUTPUT"
```

3. **Add timing**
```bash
START=$(date +%s)
# ... pipeline ...
END=$(date +%s)
echo "Completed in $((END - START)) seconds"
```

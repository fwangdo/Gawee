# Shell Scripting for full_pipeline.sh - Summary

## What You Need to Understand

### 1. Shebang Line

```bash
#!/bin/bash
```
- First line of script
- Tells OS which interpreter to use
- `/bin/bash` = Bash shell

### 2. Comments

```bash
# This is a comment
# Usage: ./script.sh arg1
```

### 3. set -e (Exit on Error)

```bash
set -e
```
- Script stops immediately if any command fails
- Without it: script continues even after errors
- **Best practice** for reliable scripts

### 4. Variables

```bash
# Assignment (NO spaces around =)
MY_VAR="hello"
LLVM_DIR="$HOME/llvm-install"

# Usage (with $)
echo $MY_VAR
echo "$MY_VAR"    # Safer - handles spaces
echo "${MY_VAR}"  # Explicit boundary
```

### 5. Special Variables

| Variable | Meaning |
|----------|---------|
| `$0` | Script name |
| `$1` | First argument |
| `$2` | Second argument |
| `$#` | Number of arguments |
| `$@` | All arguments |
| `$HOME` | User's home directory |

### 6. Command Substitution

```bash
# Capture command output into variable
RESULT=$(some_command)
RESULT=$(command1 | command2)

# Old syntax (backticks) - avoid
RESULT=`some_command`
```

### 7. Getting Script's Directory

```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
```

Breaking it down:
- `$0` = script path as invoked
- `dirname "$0"` = directory part of path
- `cd ... && pwd` = change to dir and print absolute path
- `$(...)` = capture the output

### 8. Conditionals

```bash
# Check if variable is empty
if [ -z "$VAR" ]; then
  echo "VAR is empty"
  exit 1
fi

# Check if file exists
if [ -f "$FILE" ]; then
  echo "File exists"
fi
```

| Test | Meaning |
|------|---------|
| `-z "$VAR"` | Variable is empty |
| `-n "$VAR"` | Variable is not empty |
| `-f "$FILE"` | File exists |
| `-d "$DIR"` | Directory exists |

### 9. Piping

```bash
# Output of command1 becomes input of command2
command1 | command2 | command3

# Example: chain transformations
echo "$STEP1" | mlir-opt --pass1 | mlir-opt --pass2
```

### 10. Quoting Rules

```bash
# Double quotes: variables expanded
echo "$HOME"        # /Users/hdy

# Single quotes: literal string
echo '$HOME'        # $HOME

# Always quote variables to handle spaces
FILE="my file.txt"
cat "$FILE"         # Correct
cat $FILE           # Wrong - becomes: cat my file.txt
```

### 11. Exit Codes

```bash
exit 0    # Success
exit 1    # General error

# Check previous command's exit code
if [ $? -ne 0 ]; then
  echo "Previous command failed"
fi
```

## The full_pipeline.sh Pattern

```bash
#!/bin/bash
set -e                           # Stop on errors

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL="$SCRIPT_DIR/build/my-tool"

# Get input argument
INPUT="$1"
if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input>"
  exit 1
fi

# Chain commands with pipes
STEP1=$("$TOOL" --pass1 "$INPUT")
STEP2=$(echo "$STEP1" | other-tool --pass2)
echo "$STEP2"
```

## Key Takeaways

1. **Always quote variables**: `"$VAR"` not `$VAR`
2. **Use `set -e`**: Stop on first error
3. **`$(...)`**: Capture command output
4. **`|` (pipe)**: Chain command output → input
5. **`$1`, `$2`, etc.**: Access script arguments

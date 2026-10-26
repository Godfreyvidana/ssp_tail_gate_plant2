lines = open('app_vision.py', encoding='utf-8').readlines()
# Find and fix the broken print line
for i, line in enumerate(lines):
    if 'seq1_capture_delay_secs' in line and 'print' in line and '\\\"' in line:
        lines[i] = '                print(f"[CONFIG] seq1_capture_delay_secs={SEQ1_CAPTURE_DELAY_SECS}s  progress_bar_duration_secs={PROGRESS_BAR_DURATION_SECS}s")\n'
        print(f"Fixed line {i+1}")
        break
with open('app_vision.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Done")

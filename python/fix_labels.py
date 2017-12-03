'''
0 - alex
1 - chris
2 - eben
3 - none
'''

in_filename = "backup/speaker-data-none-2.csv"
out_filename = "backup/speaker-data-none-2-correctlabels.csv"
correct_label = "3.000000000000000000e+00"

with open(out_filename, 'a') as out:
    with open(in_filename, "r") as lines:
        for line in lines:
            column_values = line.strip().split(',')
            column_values[-1] = correct_label
            new_line_string = ','.join(column_values)
            out.write(new_line_string + '\n')

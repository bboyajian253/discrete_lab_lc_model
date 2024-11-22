import my_toolbox as tb
import os

def compile_all_tex_in( folder_path: str) -> None:
    """
    compile all tex files in the given folder
    """
    for file in os.listdir(folder_path):
        if file.endswith(".tex"):
            name_no_ext = file.split('.')[0]
            tb.tex_to_pdf(folder_path + file, name_no_ext)
    return None

def make_standalone_tex(old_file_path: str, new_file_path: str = None) -> None:
    if new_file_path is None:
        new_file_path = old_file_path
    with open(old_file_path, 'r') as old_file:
        lines = old_file.readlines()
    with open(new_file_path, 'w') as new_file:
        new_file.write("\\documentclass[border=3mm,preview]{standalone} \n")
        new_file.write("\\usepackage{booktabs} \n")
        new_file.write("\\usepackage{caption} \n")
        new_file.write("\\usepackage{pdflscape} \n")
        new_file.write("\\begin{document}\n")
        for line in lines:
            new_file.write(line)
        new_file.write("\\end{document}\n")
    # close the files
    old_file.close()
    new_file.close()

def make_standalone_tex_lscape(old_file_path: str, new_file_path: str = None) -> None:
    if new_file_path is None:
        new_file_path = old_file_path
    with open(old_file_path, 'r') as old_file:
        lines = old_file.readlines()
    with open(new_file_path, 'w') as new_file:
        new_file.write("\\documentclass[border=3mm,preview]{standalone} \n")
        new_file.write("\\usepackage{booktabs} \n")
        new_file.write("\\usepackage{caption} \n")
        new_file.write("\\usepackage{pdflscape} \n")
        new_file.write("\\begin{document}\n")
        new_file.write("\\begin{landscape}\n")
        for line in lines:
            new_file.write(line)
        new_file.write("\\end{landscape}\n")
        new_file.write("\\end{document}\n")
    # close the files
    old_file.close()
    new_file.close()

def add_beg_table(old_file_path: str, new_file_path: str = None, tabular_modifier: str = None) -> None:
    if new_file_path is None:
        new_file_path = old_file_path
    with open(old_file_path, 'r') as old_file:
        lines = old_file.readlines()
    with open(new_file_path, 'w') as new_file:
        new_line = "\\begin{tabular}"
        if tabular_modifier:
            new_line += "{" + tabular_modifier + "}"
        new_line += "\n"
        new_file.write(new_line)
        for line in lines:
            new_file.write(line)
    # close the files
    old_file.close()
    new_file.close()

def add_footnote(old_file_path: str, footnote: str, new_file_path: str = None, num_cols: int = None) -> None:
    if new_file_path is None:
        new_file_path = old_file_path
    with open(old_file_path, 'r') as old_file:
        lines = old_file.readlines()
    with open(new_file_path, 'w') as new_file:
        for line in lines:
            new_file.write(line)
        # \multicolumn{num_cols}{l}{\small *THIS IS A NICE FOOTNOTE.} \\
        new_line = "\\footnotesize{" + footnote + "} \\\ \n"
        if num_cols:
            new_line = "\\multicolumn{" + str(num_cols) + "}{l}{\small{" + footnote + "}} \\\ \n"
        new_file.write(new_line)
    # close the files
    old_file.close()
    new_file.close()

def add_end_table(old_file_path: str, new_file_path: str = None) -> None:
    if new_file_path is None:
        new_file_path = old_file_path
    with open(old_file_path, 'r') as old_file:
        lines = old_file.readlines()
    with open(new_file_path, 'w') as new_file:
        for line in lines:
            new_file.write(line)
        new_file.write("\\end{tabular}\n")

    # close the files
    old_file.close()
    new_file.close()


if __name__ == "__main__":
    quant_out_path = "../do_files/UKHLS_quants_output/"
    
    old_name = "reg_results_specEdMH.tex"
    new_name = "reg_results_specEdMH_standalone.tex"
    qreg_tex_path = quant_out_path + old_name
    new_qreg_tex_path = quant_out_path + new_name
    make_standalone_tex_lscape(qreg_tex_path, new_qreg_tex_path)
    tb.tex_to_pdf(quant_out_path, new_name)

    old_name = "reg_results_Q5_specEd.tex"
    new_name = "reg_results_Q5_specEd_standalone.tex"
    qreg_tex_path = quant_out_path + old_name
    new_qreg_tex_path = quant_out_path + new_name
    make_standalone_tex_lscape(qreg_tex_path, new_qreg_tex_path)
    tb.tex_to_pdf(quant_out_path, new_name)

    old_name = "reg_results_specCont.tex"
    new_name = "reg_results_specCont_standalone.tex"
    qreg_tex_path = quant_out_path + old_name
    new_qreg_tex_path = quant_out_path + new_name
    make_standalone_tex_lscape(qreg_tex_path, new_qreg_tex_path)
    tb.tex_to_pdf(quant_out_path, new_name)

    # old_name = "reg_results_specCont_both.tex"
    # new_name = "reg_results_specCont_both_standalone.tex"
    # qreg_tex_path = quant_out_path + old_name
    # new_qreg_tex_path = quant_out_path + new_name
    # make_standalone_tex_lscape(qreg_tex_path, new_qreg_tex_path)
    # tb.tex_to_pdf(quant_out_path, new_name)


    # old_name = "reg_results_Q5_specEd_both.tex"
    # new_name = "reg_results_Q5_specEd_both_test.tex"
    # qreg_tex_path = quant_out_path + old_name
    # new_qreg_tex_path = quant_out_path + new_name
    # make_standalone_tex_lscape(qreg_tex_path, new_qreg_tex_path)
    # tb.tex_to_pdf(quant_out_path, new_name)


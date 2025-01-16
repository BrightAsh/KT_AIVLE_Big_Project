import os, sys
sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver1 as mo

contract_name = 'example.hwp'
mo.initialize_models()
indentification_results, summary_results = mo.pipline(contract_name)
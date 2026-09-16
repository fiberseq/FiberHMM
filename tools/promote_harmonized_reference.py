"""Extract the accepted event-family kernels without paper paths or CLI code.

This is a maintainer tool, not a runtime dependency. Only the transitive symbol
dependencies of the listed entry points are copied. Function bodies retain the
same AST, apart from package-relative imports. A manifest binds every source.
Run with --check to check an existing extraction without writing anything.
"""
import argparse
import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

RELEASE = Path(__file__).resolve().parents[1]
SOURCE = RELEASE.parent / 'paper/analysis/benchmark/napa_external_footprint_comparison_20260911'
DEST = RELEASE / 'fiberhmm/inference/consensus/harmonized_families/reference'
ROOTS = {
    'run_native_locus_map': ['prepare_input', 'summarize'],
    'run_bounded_parent_panel': ['evaluate_cohort', 'call_key'],
    'cross_source_family_consolidation': ['combine_cases', 'foreign_child_scores', 'cross_annotation'],
    'fitted_geometry_nomination': ['source_density_cells', 'nominate_fitted_parents'],
    'run_native_cell_consolidation': ['candidate_inputs'],
    'overlapping_family_update': ['extend_parent'],
    'reuse_consensus_fits': ['reuse_consensus'],
    'resolve_consensus_representatives': ['resolve_representatives'],
    'native_cell_consolidation': ['nominate_parents'],
}


def bindings(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name) for a in node.names]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
    return []


def extract():
    trees = {}; indices = {}; selected = defaultdict(set)
    def require(module, name):
        if module not in trees:
            trees[module] = ast.parse((SOURCE / (module+'.py')).read_text())
            indices[module] = {name: i for i, node in enumerate(trees[module].body) for name in bindings(node)}
        if name not in indices[module]:
            raise ValueError(f'Unresolved reference symbol {module}.{name}')
        index = indices[module][name]
        if index in selected[module]: return
        selected[module].add(index)
        node = trees[module].body[index]
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in indices[module]:
                require(module, child.id)
            if isinstance(child, ast.ImportFrom) and child.level == 0 and (SOURCE / ((child.module or '')+'.py')).exists():
                for alias in child.names: require(child.module, alias.name)
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if (SOURCE / (alias.name+'.py')).exists():
                        raise ValueError('Explicit symbol imports required: '+alias.name)
    for module, names in ROOTS.items():
        for name in names: require(module, name)

    class RelativeImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level == 0 and node.module in selected:
                node.level = 1
            return node
    files = {}; manifest = {}
    for module, indices_ in sorted(selected.items()):
        nodes = [deepcopy(trees[module].body[i]) for i in sorted(indices_)]
        original = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
        tree = RelativeImports().visit(ast.Module(body=nodes, type_ignores=[]))
        text = '# Extracted reference kernels; see SOURCE_MANIFEST.json.\n' + ast.unparse(tree)+'\n'
        if 'sys.path' in text or "'/mnt/" in text or 'BASE.parents' in text:
            raise ValueError('Nonportable runtime dependency in '+module)
        files[module+'.py'] = text
        manifest[module] = dict(source_sha256=hashlib.sha256((SOURCE/(module+'.py')).read_bytes()).hexdigest(),
            selected_symbols=sorted({name for node in nodes for name in bindings(node)}),
            original_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
            extracted_sha256=hashlib.sha256(text.encode()).hexdigest())
    files['__init__.py'] = '"""Portable accepted event-local fitting and consolidation kernels."""\n'
    files['SOURCE_MANIFEST.json'] = json.dumps(manifest, indent=2, sort_keys=True)+'\n'
    return files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    files = extract()
    for name, content in files.items():
        path = DEST/name
        if path.exists():
            if path.read_text() != content: raise ValueError('Existing extraction differs: '+str(path))
        elif args.check: raise FileNotFoundError(path)
        else:
            patch = '*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())+'*** End Patch\n'
            subprocess.run(['apply_patch', patch], check=True, capture_output=True)
    print(f'{len(files)-2} reference modules verified; no benchmark runtime imports')


if __name__ == '__main__': main()

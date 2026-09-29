import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_system_injects_canonical_retrievers_into_hybrid():
    source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = _class_method(tree, "QASPERRetrievalSystem", "__init__")

    hybrid_calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HybridRetriever"
    ]
    assert len(hybrid_calls) == 1
    call = hybrid_calls[0]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    bm25 = keywords["bm25_retriever"]
    embedding = keywords["embedding_retriever"]
    assert isinstance(bm25, ast.Attribute)
    assert isinstance(bm25.value, ast.Name) and bm25.value.id == "self"
    assert bm25.attr == "bm25_retriever"
    assert isinstance(embedding, ast.Attribute)
    assert isinstance(embedding.value, ast.Name) and embedding.value.id == "self"
    assert embedding.attr == "embedding_retriever"


def test_hybrid_constructor_exposes_injection_points():
    source = (
        ROOT / "src" / "retrieval" / "hybrid_retriever.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = _class_method(tree, "HybridRetriever", "__init__")
    arg_names = [arg.arg for arg in init.args.args]
    assert "bm25_retriever" in arg_names
    assert "embedding_retriever" in arg_names


def test_system_constructs_each_primary_retriever_once():
    source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = _class_method(tree, "QASPERRetrievalSystem", "__init__")

    constructor_counts = {"BM25Retriever": 0, "EmbeddingRetriever": 0}
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in constructor_counts
        ):
            constructor_counts[node.func.id] += 1

    assert constructor_counts == {
        "BM25Retriever": 1,
        "EmbeddingRetriever": 1,
    }

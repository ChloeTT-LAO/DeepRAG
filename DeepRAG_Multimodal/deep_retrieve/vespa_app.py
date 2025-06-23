from vespa.package import Schema, Document, Field, FieldSet, RankProfile, HNSW
from vespa.package import ApplicationPackage, Function, FirstPhaseRanking, SecondPhaseRanking

# 创建PDF页面架构
pdf_page_schema = Schema(
    name="pdf_page",
    document=Document(
        fields=[
            Field(
                name="id",
                type="string",
                indexing=["summary", "index"],
                match=["word"]
            ),
            Field(
                name="pdf_path",
                type="string",
                indexing=["summary", "index"]
            ),
            Field(
                name="page_index",
                type="int",
                indexing=["summary", "attribute"]
            ),
            Field(
                name="text",
                type="string",
                indexing=["index"],
                match=["text"],
                index="enable-bm25"
            ),
            Field(
                name="text_embedding",
                type="tensor<float>(x[384])",
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="angular",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=200,
                )
            ),
            Field(
                name="image_embedding",
                type="tensor<float>(x[768])",  # ColQwen2.5的向量维度
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="angular",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=200,
                )
            ),
            # 可选：添加二值化嵌入字段以节省空间
            Field(
                name="binary_text_embedding",
                type="tensor<int8>(x[48])",  # 384/8 = 48
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                )
            ),
            Field(
                name="binary_image_embedding",
                type="tensor<int8>(x[96])",  # 768/8 = 96
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                )
            )
        ]
    ),
    fieldsets=[FieldSet(name="default", fields=["text"])],
)

# 创建排名配置文件
# 1. 混合模式 - 结合文本BM25和向量相似度
mixed_rank_profile = RankProfile(
    name="mixed",
    inputs=[
        ("query(q_text_embedding)", "tensor<float>(x[384])"),
        ("query(q_image_embedding)", "tensor<float>(x[768])"),
        ("query(text_weight)", "float"),
    ],
    functions=[
        Function(
            name="textSimilarity",
            expression="closeness(field, text_embedding)",
        ),
        Function(
            name="imageSimilarity",
            expression="closeness(field, image_embedding)",
        ),
        Function(
            name="combinedScore",
            expression="query(text_weight) * textSimilarity + (1 - query(text_weight)) * imageSimilarity",
        ),
    ],
    first_phase=FirstPhaseRanking(expression="bm25(text) + combinedScore"),
    second_phase=SecondPhaseRanking(expression="combinedScore", rerank_count=100),
)

# 2. 二进制模式 - 使用汉明距离检索，适合大规模数据
binary_rank_profile = RankProfile(
    name="binary",
    inputs=[
        ("query(q_bin_text_embedding)", "tensor<int8>(x[48])"),
        ("query(q_bin_image_embedding)", "tensor<int8>(x[96])"),
        ("query(text_weight)", "float"),
    ],
    functions=[
        Function(
            name="textBinarySimilarity",
            expression="1 / (1 + hamming_distance(field, binary_text_embedding))",
        ),
        Function(
            name="imageBinarySimilarity",
            expression="1 / (1 + hamming_distance(field, binary_image_embedding))",
        ),
        Function(
            name="combinedBinaryScore",
            expression="query(text_weight) * textBinarySimilarity + (1 - query(text_weight)) * imageBinarySimilarity",
        ),
    ],
    first_phase=FirstPhaseRanking(expression="combinedBinaryScore"),
    second_phase=SecondPhaseRanking(expression="combinedBinaryScore", rerank_count=100),
)

# 添加排名配置文件到架构
pdf_page_schema.add_rank_profile(mixed_rank_profile)
pdf_page_schema.add_rank_profile(binary_rank_profile)

# 创建应用包
vespa_app = ApplicationPackage(
    name="pdf-multimodal-search",
    schema=[pdf_page_schema]
)
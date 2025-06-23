"""
Vespa retriever optimized for default Vespa configuration.
This implementation works with the default Vespa config without requiring custom app deployment.
"""

import os
import json
import time
import asyncio
import logging
import requests
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import torch
import traceback

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("vespa_retriever模块初始化")
print(f"当前模块的日志器名称: {logger.name}")


class VespaConfig:
    """Configuration for Vespa retriever"""

    def __init__(self,
                 endpoint="http://localhost:19071",
                 namespace="multimodal",
                 document_type="multimodal_document",
                 text_field="text",
                 image_field="image_embedding",
                 text_embedding_field="text_embedding",
                 verify_ssl=False,
                 timeout=15):
        self.endpoint = endpoint
        self.namespace = namespace
        self.document_type = document_type
        self.text_field = text_field
        self.image_field = image_field
        self.text_embedding_field = text_embedding_field
        self.verify_ssl = verify_ssl
        self.timeout = timeout


class VespaRetriever:
    """Vespa-based retriever for multimodal document retrieval adapted for default Vespa config"""

    def __init__(
            self,
            config: VespaConfig,
            text_model: Any,
            image_model: Optional[Any] = None,
            processor: Optional[Any] = None,
            embedding_weight: float = 0.5,
            topk: int = 10
    ):
        """Initialize VespaRetriever

        Args:
            config: Vespa configuration
            text_model: Text embedding model (FlagModel)
            image_model: Image embedding model (ColQwen2_5)
            processor: Image processor (ColQwen2_5_Processor)
            embedding_weight: Weight for text embeddings (1.0 = text only, 0.0 = image only)
            topk: Number of results to return
        """
        self.config = config
        self.text_model = text_model
        self.image_model = image_model
        self.processor = processor
        self.text_embedding_weight = embedding_weight
        self.topk = topk

        # Verify Vespa connection
        self.connection_ok = self._check_vespa_connection()

        # Initialize in-memory storage for handling documents
        self.documents = []
        self.text_embeddings = []
        self.image_embeddings = []

    def _index_documents_to_vespa(self, documents: List[Dict[str, Any]]):
        """改进的文档索引方法"""
        # 清除所有现有文档
        try:
            # 可选：在重新索引前删除所有文档
            clear_url = f"{self.config.endpoint}/document/v1/"
            requests.delete(clear_url, params={"cluster": self.config.namespace}, timeout=5)
            logger.info("已清除Vespa中的所有文档")
        except Exception as e:
            logger.warning(f"清除文档时出错: {str(e)}")

        # 索引新文档
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            text = doc.get("text", "")
            if not text.strip():
                logger.warning(f"跳过空文本文档 {doc_id}")
                continue

            text_embedding = self._compute_text_embedding(text)

            # 准备文档数据
            doc_data = {
                "fields": {
                    self.config.text_field: text,
                    self.config.text_embedding_field: {"values": text_embedding},
                    "page_index": doc.get("metadata", {}).get("page_index", 0),
                    "pdf_path": doc.get("metadata", {}).get("pdf_path", "")
                }
            }

            # 添加图像嵌入
            if self.image_model and "image" in doc and doc["image"] is not None:
                image_embedding = self._compute_image_embedding(doc["image"])
                doc_data["fields"][self.config.image_field] = {"values": image_embedding}

            # 发送到Vespa并验证
            try:
                url = f"{self.config.endpoint}/document/v1/{self.config.namespace}/{self.config.document_type}/docid/{doc_id}"
                response = requests.post(url, json=doc_data, timeout=self.config.timeout)

                if response.status_code == 200:
                    logger.info(f"成功索引文档 {doc_id}")
                else:
                    logger.warning(f"索引文档 {doc_id} 失败: {response.status_code} - {response.text[:150]}")
            except Exception as e:
                logger.error(f"索引文档 {doc_id} 出错: {str(e)}")

        # 索引完成后等待文档可用
        time.sleep(2)
        logger.info(f"索引完成 - 共 {len(documents)} 个文档")

        # 验证索引
        try:
            verify_url = f"{self.config.endpoint}/search/"
            verify_data = {"yql": f'select * from sources * where sddocname contains "{self.config.document_type}"',
                           "hits": 1000}
            response = requests.post(verify_url, json=verify_data, timeout=self.config.timeout)

            if response.status_code == 200:
                hits = response.json().get("root", {}).get("children", [])
                logger.info(f"验证索引: 找到 {len(hits)} 个文档")
            else:
                logger.warning(f"验证索引失败: {response.status_code}")
        except Exception as e:
            logger.error(f"验证索引出错: {str(e)}")

    def _check_vespa_connection(self) -> bool:
        """检查Vespa是否运行并响应"""
        try:
            response = requests.get(
                f"{self.config.endpoint}/ApplicationStatus",
                timeout=self.config.timeout,
                verify=self.config.verify_ssl  # 使用配置中的SSL验证选项
            )
            if response.status_code == 200:
                logger.info(f"✅ 成功连接到Vespa服务器: {self.config.endpoint}")
                return True
            else:
                logger.warning(f"⚠️ Vespa服务器返回状态码 {response.status_code}")
                logger.info("将使用内存向量搜索")
                return False
        except Exception as e:
            logger.error(f"❌ 无法连接到Vespa服务器: {str(e)}")
            logger.info("将使用内存向量搜索")
            return False

    def retrieve(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据查询相似性检索文档"""
        # 如果文档集改变，重新索引
        if documents != self.documents:
            self.documents = documents
            if self.connection_ok:
                self._index_documents_to_vespa(documents)
            else:
                self._precompute_embeddings(documents)

        # 计算查询嵌入
        query_embedding = self._compute_text_embedding(query)

        # 使用Vespa查询或后备使用内存检索
        if self.connection_ok:
            results = self._query_vespa(query_embedding, query)
        else:
            results = self._compute_similarity_and_rank(query_embedding)

        logger.info(f"为查询 '{query}' 检索到 {len(results)} 个文档")
        return results

    def _precompute_embeddings(self, documents: List[Dict[str, Any]]):
        """Precompute embeddings for all documents"""
        logger.info(f"Precomputing embeddings for {len(documents)} documents...")
        self.text_embeddings = []
        self.image_embeddings = []

        # 计算文本嵌入
        for doc in documents:
            text = doc.get("text", "")
            self.text_embeddings.append(self._compute_text_embedding(text))

            # 如果有图像和图像模型，计算图像嵌入
            if self.image_model and self.processor:
                image = doc.get("image", None)
                if image is not None:
                    self.image_embeddings.append(self._compute_image_embedding(image))
                else:
                    self.image_embeddings.append(None)

        # 转换为 NumPy 数组以加速计算
        self.text_embeddings = np.array(self.text_embeddings)
        logger.info("Embeddings precomputed successfully")

    def _compute_similarity_and_rank(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """计算相似度分数并排序文档"""
        scores = []

        # 确保文档和嵌入长度匹配
        if len(self.documents) != len(self.text_embeddings):
            logger.warning(f"文档数量({len(self.documents)})与嵌入数量({len(self.text_embeddings)})不匹配，重新计算嵌入")
            self._precompute_embeddings(self.documents)

        # 计算每个文档的相似度分数
        for i, doc in enumerate(self.documents):
            try:
                # 确保索引在范围内
                if i >= len(self.text_embeddings):
                    logger.warning(f"文档索引{i}超出嵌入数组范围，跳过")
                    continue

                # 文本相似度（余弦相似度）
                text_score = np.dot(query_embedding, self.text_embeddings[i])

                # 图像相似度（如果有）
                image_score = 0.0
                if self.image_model and len(self.image_embeddings) > i and self.image_embeddings[i] is not None:
                    img_emb = self.image_embeddings[i]
                    image_score = np.dot(query_embedding, img_emb) / 100  # 归一化

                # 组合分数
                combined_score = (self.text_embedding_weight * text_score +
                                  (1 - self.text_embedding_weight) * image_score)

                scores.append((i, combined_score))
            except Exception as e:
                logger.error(f"计算文档 {i} 的相似度时出错: {str(e)}")
                continue

        # 排序并选择前K个
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_scores = scores[:self.topk]

        # 格式化结果
        results = []
        for idx, score in top_k_scores:
            doc = self.documents[idx]
            results.append({
                "text": doc.get("text", ""),
                "score": float(score),
                "metadata": doc.get("metadata", {}),
                "page": doc.get("metadata", {}).get("page_index")
            })

        return results

    def _compute_text_embedding(self, text: str) -> list:
        """计算文本嵌入并确保维度正确"""
        target_dim = 384

        if not text.strip():
            return [0.0] * target_dim

        try:
            with torch.no_grad():
                embedding = self.text_model.encode([text])

                # 获取向量
                if isinstance(embedding, torch.Tensor):
                    embedding_vector = embedding[0].cpu().numpy()
                else:
                    embedding_vector = embedding[0]

                # 确保维度匹配
                current_dim = len(embedding_vector)

                norm = np.linalg.norm(embedding_vector)
                if norm > 0:
                    embedding_vector = embedding_vector / norm

                if current_dim > target_dim:
                    result = embedding_vector[:target_dim]
                elif current_dim < target_dim:
                    result = np.concatenate([embedding_vector, np.zeros(target_dim - current_dim)])
                else:
                    result = embedding_vector

                # 关键步骤：显式转换为Python float类型
                result = [float(x) for x in result]

                return result

        except Exception as e:
            logger.error(f"计算文本嵌入时出错: {str(e)}")
            return [0.0] * target_dim

    def _compute_image_embedding(self, image: Image.Image) -> list:
        """计算图像嵌入并确保维度正确"""
        target_dim = 1024

        if image is None:
            return [0.0] * target_dim

        try:
            with torch.no_grad():
                query_embedding = self.image_model(**self.processor.process_queries([""]).to(self.image_model.device))
                image_input = self.processor.process_images([image]).to(self.image_model.device)
                image_embedding = self.image_model(**image_input)

                if isinstance(image_embedding, torch.Tensor):
                    image_embedding = image_embedding.cpu().numpy()

                if image_embedding.ndim > 2:
                    image_embedding = image_embedding[0]

                if image_embedding.ndim == 2:
                    flat_embedding = np.mean(image_embedding, axis=0)
                else:
                    flat_embedding = image_embedding

                current_dim = len(flat_embedding)
                norm = np.linalg.norm(flat_embedding)
                if norm > 0:
                    flat_embedding = flat_embedding / norm


                if current_dim > target_dim:
                    result = flat_embedding[:target_dim]
                elif current_dim < target_dim:
                    result = np.concatenate([flat_embedding, np.zeros(target_dim - current_dim)])
                else:
                    result = flat_embedding

                # 关键步骤：显式转换为Python float类型
                result = [float(x) for x in result]

                return result

        except Exception as e:
            logger.error(f"计算图像嵌入时出错: {str(e)}")
            return [0.0] * target_dim

        except Exception as e:
            logger.error(f"计算图像嵌入时出错: {str(e)}")
            return [0.0] * target_dim

    def _query_vespa(self, query_embedding: np.ndarray, original_query: str) -> List[Dict[str, Any]]:
        """使用向量嵌入查询Vespa"""
        try:
            # 确保查询嵌入是列表格式
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # 1. 简化YQL查询 - 移除文本匹配约束
            query_data = {
                "yql": f'select * from sources * where sddocname contains "{self.config.document_type}"',
                "hits": self.topk,
                "ranking": "default",  # 2. 使用默认排序方法，确保存在
                "timeout": "10s",  # 增加超时时间
            }

            # 3. 正确传递向量嵌入
            query_data[f"ranking.features.query({self.config.text_embedding_field})"] = query_embedding

            # 4. 添加基本文本匹配（可选）
            if original_query:
                # 使用模糊匹配而非精确匹配
                query_data["query"] = original_query

                # 记录详细的查询信息
            logger.info(f"Vespa查询类型: {query_data.get('yql')}")

            url = f"{self.config.endpoint}/search/"
            response = requests.post(
                url,
                json=query_data,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )

            if response.status_code == 200:
                results = response.json()
                hits = results.get("root", {}).get("children", [])

                # 记录返回的文档ID和得分
                logger.info(f"Vespa返回 {len(hits)} 个结果")
                for i, hit in enumerate(hits):
                    logger.info(f"结果 {i + 1}: ID={hit.get('id')}, 得分={hit.get('relevance')}")

                # 格式化结果
                formatted_results = []
                for hit in hits:
                    fields = hit.get("fields", {})
                    formatted_results.append({
                        "text": fields.get(self.config.text_field, ""),
                        "score": hit.get("relevance", 0.0),
                        "metadata": {
                            "page_index": fields.get("page_index", None),
                            "pdf_path": fields.get("pdf_path", "")
                        },
                        "id": hit.get("id", "")  # 添加文档ID便于调试
                    })

                return formatted_results
            else:
                # 详细记录错误
                logger.warning(f"Vespa查询失败，状态码: {response.status_code}, 响应: {response.text[:200]}")
                return self._compute_similarity_and_rank(query_embedding)
        except Exception as e:
            logger.error(f"查询Vespa时出错: {str(e)}")
            return self._compute_similarity_and_rank(query_embedding)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
import torch
from pdf2image import convert_from_path
from PIL import Image
import subprocess
import logging

# 创建日志目录
log_dir = Path("/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "multimodal_intent_test.log"

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 开头加入测试日志
logger.info("=== 日志系统初始化完成，日志文件: %s ===", log_file)

# 添加必要的路径
sys.path.append("DeepRAG_Multimodal/deep_retrieve")
# 加载环境变量
load_dotenv(
    "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher
from DeepRAG_Multimodal.deep_retrieve.vespa_retriever import VespaConfig, VespaRetriever
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_vespa import DeepSearch_Vespa

import os

# 在脚本开头添加这个函数
def read_pem_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None


class MultimodalIntentTester:
    """多模态多意图检索测试类"""

    def __init__(self, args=None):
        """初始化测试器"""
        self.args = args or self.parse_arguments()
        if self.args.vespa_api_key and os.path.exists(self.args.vespa_api_key):
            self.args.vespa_api_key = read_pem_file(self.args.vespa_api_key)
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)

        # 检查Vespa是否启动，如果需要初始化则启动
        if self.args.use_vespa or self.args.use_vespa_cloud:
            logger.info("Vespa模式已启用")
            if self.args.use_vespa_cloud:
                logger.info(f"将使用Vespa云，租户: {self.config['vespa_tenant']}")
            else:
                logger.info(f"将连接到Vespa端点: {self.config['vespa_endpoint']}")

        self.setup_models()

    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='多模态多意图检索测试')
        # 路径和基本配置
        parser.add_argument('--test_data', type=str,
                            default='/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl',
                            help='测试数据集路径')
        parser.add_argument('--pdf_dir', type=str,
                            default='/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc',
                            help='PDF文件目录')
        parser.add_argument('--results_dir', type=str,
                            default='test_results',
                            help='结果保存目录')
        parser.add_argument('--sample_size', type=int, default=40,
                            help='测试样本数量，0表示全部')
        parser.add_argument('--ocr_method', type=str, default='paddleocr',
                            choices=['paddleocr', 'pytesseract'],
                            help='OCR方法选择')
        parser.add_argument('--device', type=str, default='cpu',
                            help='设备选择')
        parser.add_argument('--mode', type=str, default='all',
                            choices=['weight', 'intent', 'all'],
                            help='测试模式: weight(权重对比), intent(意图对比), all(全部)')
        parser.add_argument('--retrieval_mode', type=str, default='mixed',
                            choices=['mixed', 'text_only'],
                            help='检索模式: mixed(多模态), text_only(仅文本)')

        # Vespa相关配置
        parser.add_argument('--use_vespa', default='True', action='store_true',
                            help='是否使用Vespa服务')
        parser.add_argument('--vespa_endpoint', type=str, default='http://localhost:8080',
                            help='Vespa服务器地址')
        parser.add_argument('--use_vespa_cloud', default=None,
                            help='是否使用Vespa云服务')
        parser.add_argument('--vespa_tenant', type=str, default='deepsearch',
                            help='Vespa云租户名称')
        parser.add_argument('--vespa_api_key', type=str, default='/Users/chloe/Downloads/laojt5.deepsearch.pem',
                            help='Vespa云API密钥')
        parser.add_argument('--vespa_app_name', type=str, default='multiintend',
                            help='Vespa应用名称')

        parser.add_argument('--debug', action='store_true',
                            help='是否开启调试模式')
        return parser.parse_args()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置
            'test_data_path': self.args.test_data,
            'pdf_base_dir': self.args.pdf_dir,
            'results_dir': self.args.results_dir,

            # 采样配置
            'sample_size': self.args.sample_size,
            'debug': self.args.debug,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 15,
            'rerank_topk': 10,
            'text_weight': 0.6,  # 文本权重默认值
            'image_weight': 0.4,  # 图像权重默认值

            # 模型配置
            'mm_model_name': "vidore/colqwen2.5-v0.2",
            'mm_processor_name': "vidore/colqwen2.5-v0.1",
            'bge_model_name': "BAAI/bge-large-en-v1.5",
            'device': self.args.device,
            'batch_size': 2,
            'retrieval_mode': self.args.retrieval_mode,
            'ocr_method': self.args.ocr_method,

            # Vespa配置
            'use_vespa': self.args.use_vespa,
            'vespa_endpoint': self.args.vespa_endpoint,
            'use_vespa_cloud': self.args.use_vespa_cloud,
            'vespa_tenant': self.args.vespa_tenant,
            'vespa_api_key': self.args.vespa_api_key,
            'vespa_app_name': self.args.vespa_app_name
        }

        if config['debug']:
            config['sample_size'] = 2  # 调试模式下只测试2个样本

        return config

    def setup_models(self):
        """Initialize retrieval models"""
        logger.info("Initializing retrieval models...")

        # Initialize reranker
        self.reranker = FlagReranker(model_name_or_path="BAAI/bge-reranker-large")

        # Initialize embedding models
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from FlagEmbedding import FlagModel

        self.text_model = FlagModel(
            self.config['bge_model_name'],
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,
            device=self.config['device']
        )

        if self.config['retrieval_mode'] == 'mixed':
            self.image_model = ColQwen2_5.from_pretrained(
                self.config['mm_model_name'],
                torch_dtype=torch.float16,
                device_map=self.config['device']
            ).eval()

            self.processor = ColQwen2_5_Processor.from_pretrained(
                self.config['mm_processor_name'],
                size={"shortest_edge": 512, "longest_edge": 1024}
            )
        else:
            self.image_model = None
            self.processor = None

        # Set up Vespa retriever
        if self.config['use_vespa'] and self.config['vespa_endpoint']:
            logger.info(f"Using existing Vespa service for retrieval: {self.config['vespa_endpoint']}")

            self.retriever = DeepSearch_Vespa(
                text_model=self.text_model,
                image_model=self.image_model,
                processor=self.processor,
                max_iterations=self.config['max_iterations'],
                reranker=self.reranker,
                params={
                    "embedding_topk": self.config['embedding_topk'],
                    "rerank_topk": self.config['rerank_topk'],
                    "text_weight": self.config['text_weight'],
                    "image_weight": self.config['image_weight']
                },
                vespa_endpoint=self.config['vespa_endpoint']
            )

            # # Index all PDFs from the directory
            # logger.info("Indexing all PDFs from directory...")
            # self.retriever.index_all_pdfs_from_directory(
            #     self.config['pdf_base_dir'],
            #     ocr_method=self.config['ocr_method']
            # )
            # logger.info("PDF indexing complete")

        else:
            logger.info("Using standard retrieval method")

            # Initialize multimodal matcher configuration
            retriever_config = RetrieverConfig(
                model_name=self.config['mm_model_name'],
                processor_name=self.config['mm_processor_name'],
                bge_model_name=self.config['bge_model_name'],
                device=self.config['device'],
                use_fp16=True,
                batch_size=self.config['batch_size'],
                mode=self.config['retrieval_mode'],
                ocr_method=self.config['ocr_method']
            )

            # Use standard MultimodalMatcher
            self.mm_matcher = MultimodalMatcher(
                config=retriever_config,
                embedding_weight=self.config['text_weight'],
                topk=self.config['rerank_topk']
            )

            # Initialize multi-intent searcher
            self.multi_intent_search = DeepSearch_Beta(
                max_iterations=self.config['max_iterations'],
                reranker=self.reranker,
                params={
                    "embedding_topk": self.config['embedding_topk'],
                    "rerank_topk": self.config['rerank_topk'],
                    "text_weight": self.config['text_weight'],
                    "image_weight": self.config['image_weight']
                }
            )

            # Initialize single-intent searcher (disable intent breakdown)
            self.single_intent_search = DeepSearch_Beta(
                max_iterations=self.config['max_iterations'],
                reranker=self.reranker,
                params={
                    "embedding_topk": self.config['embedding_topk'],
                    "rerank_topk": self.config['rerank_topk'],
                    "text_weight": self.config['text_weight'],
                    "image_weight": self.config['image_weight']
                }
            )

            # Disable intent breakdown functions
            self.single_intent_search._split_query_intent_exist = lambda query: [query]
            self.single_intent_search._split_query_intent = lambda query: [query]
            self.single_intent_search._refine_query_intent = lambda original_query, intent_queries, context: [
                original_query]

        # For Vespa mode, set up single intent methods for testing
        if hasattr(self, 'retriever'):
            # Save original methods for multi-intent search
            self.original_split_intent = self.retriever._split_query_intent
            self.original_refine_intent = self.retriever._refine_query_intent

            # Define single-intent functions
            def single_split_intent(q):
                return [q]

            def single_refine_intent(o, i, c):
                return [o]

            # Save single-intent methods for later use
            self.single_split_intent = single_split_intent
            self.single_refine_intent = single_refine_intent

            # Use single retriever for both modes
            self.multi_intent_search = self.retriever
            self.single_intent_search = self.retriever

        logger.info("Model initialization complete")

    def load_test_data(self):
        allowed_doc_nos = [
            '4046173.pdf', '4176503.pdf', '4057524.pdf', '4064501.pdf', '4057121.pdf', '4174854.pdf',
            '4148165.pdf', '4129570.pdf', '4010333.pdf', '4147727.pdf', '4066338.pdf', '4031704.pdf',
            '4050613.pdf', '4072260.pdf', '4091919.pdf', '4094684.pdf', '4063393.pdf', '4132494.pdf',
            '4185438.pdf', '4129670.pdf', '4138347.pdf', '4190947.pdf', '4100212.pdf', '4173940.pdf',
            '4069930.pdf', '4174181.pdf', '4027862.pdf', '4012567.pdf', '4145761.pdf', '4078345.pdf',
            '4061601.pdf', '4170122.pdf', '4077673.pdf', '4107960.pdf', '4005877.pdf', '4196005.pdf',
            '4126467.pdf', '4088173.pdf', '4106951.pdf', '4086173.pdf', '4072232.pdf', '4111230.pdf',
            '4057714.pdf'
        ]
        logger.info(f"加载测试数据: {self.config['test_data_path']}")
        test_data = []

        with open(self.config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("pdf_path") in allowed_doc_nos:
                        test_data.append(item)

        if self.config['sample_size'] > 0 and len(test_data) > self.config['sample_size']:
            np.random.seed(42)  # 设置随机种子确保可重复性
            test_data = np.random.choice(test_data, self.config['sample_size'], replace=False).tolist()

        logger.info(f"成功加载 {len(test_data)} 条测试数据")
        return test_data

    def process_single_document(self, doc_data):
        """处理单个文档，使用预处理文本和PDF图像

        根据MultimodalMatcher的接口要求，格式化文档数据
        """
        documents = []

        # 获取PDF文件路径
        pdf_path = os.path.join(self.config['pdf_base_dir'], doc_data["pdf_path"])

        # try:
        #     pages = convert_from_path(pdf_path)
        #     logger.info(f"成功将PDF转换为 {len(pages)} 页图像")
        # except Exception as e:
        #     logger.error(f"转换PDF时出错：{str(e)}")
        #     return []

        # 获取预处理的OCR结果
        ocr_file = os.path.join(
            self.config['pdf_base_dir'],
            f"{self.config['ocr_method']}_save",
            f"{os.path.basename(doc_data['pdf_path']).replace('.pdf', '.json')}"
        )

        # 读取预处理的文本数据
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            logger.info(f"成功读取预处理文本文件: {ocr_file}")
        else:
            logger.warning(f"找不到预处理文本文件: {ocr_file}")
            return []

        # # 验证页面数量匹配
        # if len(loaded_data) != len(pages):
        #     logger.warning(f"OCR数据页数 ({len(loaded_data)}) 与PDF页数 ({len(pages)}) 不匹配")
        #     # 使用较小的数量
        #     page_count = min(len(loaded_data), len(pages))
        # else:
        #     page_count = len(pages)

        page_count = len(loaded_data)
        # 为每一页创建文档对象
        page_keys = list(loaded_data.keys())
        for idx in range(page_count):
            # if idx >= len(pages):
            #     break
            #
            # # 检查页面尺寸是否有效
            # page = pages[idx]
            # width, height = page.size
            # if width <= 0 or height <= 0:
            #     logger.warning(f"跳过无效页面 {idx + 1}：尺寸 {width}x{height}")
            #     continue

            # 获取OCR文本
            page_text = loaded_data[page_keys[idx]] if idx < len(page_keys) else ""

            # 创建文档结构
            documents.append({
                "text": page_text,
                # "image": page,
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"成功创建 {len(documents)} 个文档对象")
        return documents

    def test_modal_weights(self):
        """Test different modal weight configurations"""
        logger.info("Starting to test different modal weight configurations...")
        test_data = self.load_test_data()
        results = []

        # Define different weight configurations
        weight_configs = [
            {"text": 1.0, "image": 0.0, "name": "Text-only mode"},
            {"text": 0.0, "image": 1.0, "name": "Image-only mode"},
            {"text": 0.7, "image": 0.3, "name": "Text-dominant"},
            {"text": 0.5, "image": 0.5, "name": "Balanced mode"},
            {"text": 0.3, "image": 0.7, "name": "Image-dominant"}
        ]

        # Test each configuration
        for config_idx, weight in enumerate(weight_configs):
            logger.info(f"Testing config {config_idx + 1}/{len(weight_configs)}: {weight['name']}")

            # Update retriever weights
            if hasattr(self, 'retriever'):
                # Vespa mode
                self.retriever.params["text_weight"] = weight["text"]
                self.retriever.params["image_weight"] = weight["image"]
            else:
                # Standard mode
                self.multi_intent_search.params["text_weight"] = weight["text"]
                self.multi_intent_search.params["image_weight"] = weight["image"]
                self.mm_matcher.text_embedding_weight = weight["text"]

            config_results = []

            # Test each test data
            for idx, doc_data in enumerate(tqdm(test_data, desc=f"Retrieval test - {weight['name']}")):
                try:
                    # Prepare data
                    query = doc_data.get("question", "")
                    evidence_pages = doc_data.get("evidence_pages", [])
                    pdf_path = doc_data.get("pdf_path", "")

                    # For Vespa mode, search directly with the PDF path
                    if hasattr(self, 'retriever'):
                        start_time = time.time()
                        retrieval_results = self.retriever.search_pdf_retrieval(
                            query=query,
                            pdf_path=os.path.join(self.config['pdf_base_dir'], pdf_path)
                        )
                        elapsed_time = time.time() - start_time
                    else:
                        # Standard mode - process document and search
                        document_pages = self.process_single_document(doc_data)
                        if not document_pages:
                            logger.warning(f"Skipping document {pdf_path}: No valid content")
                            continue

                        data = {
                            "query": query,
                            "documents": document_pages
                        }

                        start_time = time.time()
                        retrieval_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)
                        elapsed_time = time.time() - start_time

                    # Extract page numbers from results
                    retrieved_pages = set()
                    for result in retrieval_results:
                        if 'metadata' in result and 'page_index' in result['metadata']:
                            retrieved_pages.add(result['metadata']['page_index'])
                        elif 'page' in result and result['page'] is not None:
                            retrieved_pages.add(result['page'])

                    # Evaluate results
                    evidence_set = set(evidence_pages)
                    correct_pages = evidence_set.intersection(retrieved_pages)

                    recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
                    precision = len(correct_pages) / len(retrieved_pages) if retrieved_pages else 0
                    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

                    # Record results
                    result = {
                        "doc_id": doc_data.get("doc_no", ""),
                        "pdf_path": pdf_path,
                        "query": query,
                        "weight_config": weight["name"],
                        "text_weight": weight["text"],
                        "image_weight": weight["image"],
                        "retrieved_pages": list(retrieved_pages),
                        "evidence_pages": evidence_pages,
                        "correct_pages": list(correct_pages),
                        "recall": recall,
                        "precision": precision,
                        "f1": f1,
                        "retrieval_time": elapsed_time,
                        "success": len(correct_pages) == len(evidence_set)  # Whether all evidence pages found
                    }

                    config_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing document {doc_data.get('pdf_path', '')}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

            # Calculate average metrics for current config
            if config_results:
                avg_recall = sum(r["recall"] for r in config_results) / len(config_results)
                avg_precision = sum(r["precision"] for r in config_results) / len(config_results)
                avg_f1 = sum(r["f1"] for r in config_results) / len(config_results)
                avg_time = sum(r["retrieval_time"] for r in config_results) / len(config_results)
                success_rate = sum(1 for r in config_results if r["success"]) / len(config_results) * 100

                logger.info(f"{weight['name']} average metrics: "
                            f"Recall={avg_recall:.4f}, "
                            f"Precision={avg_precision:.4f}, "
                            f"F1={avg_f1:.4f}, "
                            f"Time={avg_time:.2f}s, "
                            f"Success rate={success_rate:.2f}%")

            # Add current config results to total results
            results.extend(config_results)

        # Save results
        retriever_type = "vespa" if hasattr(self, 'retriever') else "standard"
        result_file = os.path.join(self.config['results_dir'], f'modal_weights_results_{retriever_type}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Modal weight test results saved to: {result_file}")
        return results

    def test_intent_modes(self):
        """Test single-intent vs multi-intent retrieval effects"""
        logger.info("Starting to test single-intent vs multi-intent retrieval comparison...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="Single-intent vs Multi-intent comparison test")):
            try:
                query = doc_data.get("question", "")
                evidence_pages = doc_data.get("evidence_pages", [])
                pdf_path = doc_data.get("pdf_path", "")

                # For Vespa mode
                if hasattr(self, 'retriever'):
                    # Set single-intent mode
                    self.single_intent_search._split_query_intent = self.single_split_intent
                    self.single_intent_search._refine_query_intent = self.single_refine_intent

                    document_pages = self.process_single_document(doc_data)
                    data = {
                        "query": query,
                        "documents": document_pages
                    }
                    # Test single-intent retrieval
                    single_start = time.time()
                    single_results = self.single_intent_search.search_pdf_retrieval(
                        data=data,
                        pdf_path=os.path.join(self.config['pdf_base_dir'], pdf_path)
                    )
                    single_elapsed = time.time() - single_start

                    # Restore original intent breakdown methods for multi-intent test
                    self.multi_intent_search._split_query_intent = self.original_split_intent
                    self.multi_intent_search._refine_query_intent = self.original_refine_intent

                    # # Test multi-intent retrieval
                    # multi_start = time.time()
                    # multi_results = self.multi_intent_search.search_pdf_retrieval(
                    #     data=data,
                    #     pdf_path=os.path.join(self.config['pdf_base_dir'], pdf_path)
                    # )
                    # multi_elapsed = time.time() - multi_start
                else:
                    # Standard mode - process document first
                    document_pages = self.process_single_document(doc_data)
                    if not document_pages:
                        logger.warning(f"Skipping document {pdf_path}: No valid content")
                        continue

                    data = {
                        "query": query,
                        "documents": document_pages
                    }

                    # Test single-intent retrieval
                    single_start = time.time()
                    single_results = self.single_intent_search.search_retrieval(data, retriever=self.mm_matcher)
                    single_elapsed = time.time() - single_start

                    # Test multi-intent retrieval
                    multi_start = time.time()
                    multi_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)
                    multi_elapsed = time.time() - multi_start

                # Extract page numbers from single-intent results
                single_pages = set()
                for result in single_results:
                    if 'metadata' in result and 'page_index' in result['metadata']:
                        single_pages.add(result['metadata']['page_index'])
                    elif 'page' in result and result['page'] is not None:
                        single_pages.add(result['page'])

                # # Extract page numbers from multi-intent results
                # multi_pages = set()
                # for result in multi_results:
                #     if 'metadata' in result and 'page_index' in result['metadata']:
                #         multi_pages.add(result['metadata']['page_index'])
                #     elif 'page' in result and result['page'] is not None:
                #         multi_pages.add(result['page'])

                # Evaluate results
                evidence_set = set(evidence_pages)
                single_correct = evidence_set.intersection(single_pages)
                # multi_correct = evidence_set.intersection(multi_pages)

                # Calculate metrics - single-intent
                single_recall = len(single_correct) / len(evidence_set) if evidence_set else 0
                single_precision = len(single_correct) / len(single_pages) if single_pages else 0
                single_f1 = 2 * single_recall * single_precision / (single_recall + single_precision) if (
                                                                                                                 single_recall + single_precision) > 0 else 0

                # # Calculate metrics - multi-intent
                # multi_recall = len(multi_correct) / len(evidence_set) if evidence_set else 0
                # multi_precision = len(multi_correct) / len(multi_pages) if multi_pages else 0
                # multi_f1 = 2 * multi_recall * multi_precision / (multi_recall + multi_precision) if (
                #                                                                                             multi_recall + multi_precision) > 0 else 0

                # Record results
                result = {
                    "doc_id": doc_data.get("doc_no", ""),
                    "pdf_path": pdf_path,
                    "query": query,
                    "evidence_pages": evidence_pages,
                    "task_tag": doc_data.get("task_tag", ""),
                    "subTask": doc_data.get("subTask", []),
                    "single_intent": {
                        "retrieved_pages": list(single_pages),
                        "correct_pages": list(single_correct),
                        "recall": single_recall,
                        "precision": single_precision,
                        "f1": single_f1,
                        "retrieval_time": single_elapsed,
                        "success": len(single_correct) == len(evidence_set)
                    },
                    # "multi_intent": {
                    #     "retrieved_pages": list(multi_pages),
                    #     "correct_pages": list(multi_correct),
                    #     "recall": multi_recall,
                    #     "precision": multi_precision,
                    #     "f1": multi_f1,
                    #     "retrieval_time": multi_elapsed,
                    #     "success": len(multi_correct) == len(evidence_set)
                    # }
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing document {doc_data.get('pdf_path', '')}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        # Save results
        retriever_type = "vespa" if hasattr(self, 'retriever') else "standard"
        result_file = os.path.join(self.config['results_dir'], f'intent_comparison_results_{retriever_type}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Analyze results
        self.analyze_intent_results(results)

        logger.info(f"Intent mode comparison test results saved to: {result_file}")
        return results

    def analyze_intent_results(self, results):
        """分析并打印意图对比测试结果"""
        if not results:
            logger.warning("没有可用的结果进行分析")
            return

        # 计算平均指标
        single_recalls = [r["single_intent"]["recall"] for r in results]
        # multi_recalls = [r["multi_intent"]["recall"] for r in results]

        single_precisions = [r["single_intent"]["precision"] for r in results]
        # multi_precisions = [r["multi_intent"]["precision"] for r in results]

        single_f1s = [r["single_intent"]["f1"] for r in results]
        # multi_f1s = [r["multi_intent"]["f1"] for r in results]

        single_times = [r["single_intent"]["retrieval_time"] for r in results]
        # multi_times = [r["multi_intent"]["retrieval_time"] for r in results]

        single_success = sum(1 for r in results if r["single_intent"]["success"])
        # multi_success = sum(1 for r in results if r["multi_intent"]["success"])

        # 计算平均值
        avg_single_recall = np.mean(single_recalls)
        # avg_multi_recall = np.mean(multi_recalls)

        avg_single_precision = np.mean(single_precisions)
        # avg_multi_precision = np.mean(multi_precisions)

        avg_single_f1 = np.mean(single_f1s)
        # avg_multi_f1 = np.mean(multi_f1s)

        avg_single_time = np.mean(single_times)
        # avg_multi_time = np.mean(multi_times)

        single_success_rate = single_success / len(results) * 100
        # multi_success_rate = multi_success / len(results) * 100

        # # 计算提升幅度
        # recall_improvement = (
        #                                  avg_multi_recall - avg_single_recall) / avg_single_recall * 100 if avg_single_recall > 0 else float(
        #     'inf')
        # precision_improvement = (
        #                                     avg_multi_precision - avg_single_precision) / avg_single_precision * 100 if avg_single_precision > 0 else float(
        #     'inf')
        # f1_improvement = (avg_multi_f1 - avg_single_f1) / avg_single_f1 * 100 if avg_single_f1 > 0 else float('inf')
        # time_increase = (avg_multi_time - avg_single_time) / avg_single_time * 100 if avg_single_time > 0 else float(
        #     'inf')
        # success_improvement = multi_success_rate - single_success_rate

        # 打印结果
        retriever_type = "Vespa" if hasattr(self, 'retriever') else "标准"
        logger.info(f"\n============ {retriever_type}检索: 单意图 vs 多意图检索性能分析 ============")
        logger.info(f"测试文档数: {len(results)}")
        logger.info("\n单意图检索性能:")
        logger.info(f"  平均召回率: {avg_single_recall:.4f}")
        logger.info(f"  平均精确率: {avg_single_precision:.4f}")
        logger.info(f"  平均F1值: {avg_single_f1:.4f}")
        logger.info(f"  平均检索时间: {avg_single_time:.2f}秒")
        logger.info(f"  成功率: {single_success_rate:.2f}% ({single_success}/{len(results)})")

        # logger.info("\n多意图检索性能:")
        # logger.info(f"  平均召回率: {avg_multi_recall:.4f}")
        # logger.info(f"  平均精确率: {avg_multi_precision:.4f}")
        # logger.info(f"  平均F1值: {avg_multi_f1:.4f}")
        # logger.info(f"  平均检索时间: {avg_multi_time:.2f}秒")
        # logger.info(f"  成功率: {multi_success_rate:.2f}% ({multi_success}/{len(results)})")
        #
        # logger.info("\n性能提升:")
        # logger.info(f"  召回率提升: {recall_improvement:.2f}%")
        # logger.info(f"  精确率提升: {precision_improvement:.2f}%")
        # logger.info(f"  F1值提升: {f1_improvement:.2f}%")
        # logger.info(f"  成功率提升: {success_improvement:.2f}%")
        # logger.info(f"  检索时间增加: {time_increase:.2f}%")

        # # 按任务类型分析
        # task_types = {}
        # for r in results:
        #     task_tag = r.get("task_tag", "Unknown")
        #
        #     if task_tag not in task_types:
        #         task_types[task_tag] = {
        #             "count": 0,
        #             "single_f1": 0,
        #             "multi_f1": 0,
        #             "single_success": 0,
        #             "multi_success": 0
        #         }
        #
        #     task_types[task_tag]["count"] += 1
        #     task_types[task_tag]["single_f1"] += r["single_intent"]["f1"]
        #     task_types[task_tag]["multi_f1"] += r["multi_intent"]["f1"]
        #     task_types[task_tag]["single_success"] += 1 if r["single_intent"]["success"] else 0
        #     task_types[task_tag]["multi_success"] += 1 if r["multi_intent"]["success"] else 0
        #
        # logger.info("\n按任务类型分析:")
        # for task_tag, stats in task_types.items():
        #     count = stats["count"]
        #     avg_single_f1 = stats["single_f1"] / count
        #     avg_multi_f1 = stats["multi_f1"] / count
        #     single_success_rate = stats["single_success"] / count * 100
        #     multi_success_rate = stats["multi_success"] / count * 100
        #     f1_improvement = (avg_multi_f1 - avg_single_f1) / avg_single_f1 * 100 if avg_single_f1 > 0 else float('inf')
        #     success_improvement = multi_success_rate - single_success_rate
        #
        #     logger.info(f"\n  {task_tag} (样本数: {count}):")
        #     logger.info(f"    单意图 F1: {avg_single_f1:.4f}, 成功率: {single_success_rate:.2f}%")
        #     logger.info(f"    多意图 F1: {avg_multi_f1:.4f}, 成功率: {multi_success_rate:.2f}%")
        #     logger.info(f"    F1提升: {f1_improvement:.2f}%, 成功率提升: {success_improvement:.2f}%")
        #
        # logger.info("\n=================================================")
        #
        # # 创建可视化图表
        # self.visualize_intent_results(results)

    def visualize_intent_results(self, results):
        """可视化意图对比测试结果"""
        try:
            # 创建可视化目录
            vis_dir = os.path.join(self.config['results_dir'], 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 提取指标
            metrics = {
                "单意图": {
                    "召回率": np.mean([r["single_intent"]["recall"] for r in results]),
                    "精确率": np.mean([r["single_intent"]["precision"] for r in results]),
                    "F1值": np.mean([r["single_intent"]["f1"] for r in results])
                },
                "多意图": {
                    "召回率": np.mean([r["multi_intent"]["recall"] for r in results]),
                    "精确率": np.mean([r["multi_intent"]["precision"] for r in results]),
                    "F1值": np.mean([r["multi_intent"]["f1"] for r in results])
                }
            }

            # 1. 性能对比柱状图
            plt.figure(figsize=(10, 6))
            metric_names = ["召回率", "精确率", "F1值"]
            x = np.arange(len(metric_names))
            width = 0.35

            plt.bar(x - width / 2, [metrics["单意图"][m] for m in metric_names], width, label='单意图')
            plt.bar(x + width / 2, [metrics["多意图"][m] for m in metric_names], width, label='多意图')

            plt.xlabel('评价指标')
            plt.ylabel('得分')
            plt.title('单意图 vs 多意图检索性能对比')
            plt.xticks(x, metric_names)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.savefig(os.path.join(vis_dir, 'intent_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 检索时间对比
            avg_single_time = np.mean([r["single_intent"]["retrieval_time"] for r in results])
            avg_multi_time = np.mean([r["multi_intent"]["retrieval_time"] for r in results])

            plt.figure(figsize=(8, 5))
            plt.bar(['单意图', '多意图'], [avg_single_time, avg_multi_time])
            plt.ylabel('平均检索时间 (秒)')
            plt.title('单意图 vs 多意图检索时间对比')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')

            plt.savefig(os.path.join(vis_dir, 'intent_time_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 按任务类型分析
            task_types = {}
            for r in results:
                task_tag = r.get("task_tag", "Unknown")

                if task_tag not in task_types:
                    task_types[task_tag] = {
                        "count": 0,
                        "single_f1": 0,
                        "multi_f1": 0
                    }

                task_types[task_tag]["count"] += 1
                task_types[task_tag]["single_f1"] += r["single_intent"]["f1"]
                task_types[task_tag]["multi_f1"] += r["multi_intent"]["f1"]

            if task_types:
                plt.figure(figsize=(12, 6))
                task_names = list(task_types.keys())
                x = np.arange(len(task_names))
                width = 0.35

                single_f1s = [task_types[t]["single_f1"] / task_types[t]["count"] for t in task_names]
                multi_f1s = [task_types[t]["multi_f1"] / task_types[t]["count"] for t in task_names]

                plt.bar(x - width / 2, single_f1s, width, label='单意图')
                plt.bar(x + width / 2, multi_f1s, width, label='多意图')

                plt.xlabel('任务类型')
                plt.ylabel('平均F1值')
                plt.title('不同任务类型下的检索性能对比')
                plt.xticks(x, task_names, rotation=45, ha='right')
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                plt.savefig(os.path.join(vis_dir, 'task_type_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()

            logger.info(f"可视化结果已保存到: {vis_dir}")

        except Exception as e:
            logger.error(f"创建可视化图表时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def run(self):
        """运行测试"""
        logger.info("开始多模态多意图检索测试...")
        start_time = time.time()

        try:
            test_mode = self.args.mode.lower()

            # if test_mode in ['weight', 'all']:
            #     self.test_modal_weights()

            if test_mode in ['intent', 'all']:
                self.test_intent_modes()

            total_time = time.time() - start_time
            logger.info(f"测试完成，总耗时: {total_time:.2f}秒")
            logger.info(f"结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"测试过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    # 创建测试器并运行
    tester = MultimodalIntentTester()
    tester.run()


if __name__ == "__main__":
    main()
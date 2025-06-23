import os
import socket
import subprocess
import sys
import time
import traceback

sys.path.append(
    "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")
import requests
import torch
import json
import numpy as np
from PIL import Image
from vespa.deployment import VespaCloud, VespaDocker
from DeepRAG_Multimodal.deep_retrieve.ming.agent_gpt4 import AzureGPT4Chat, create_response_format
from vespa.package import ApplicationPackage, Function, FirstPhaseRanking, SecondPhaseRanking
from vespa.application import Vespa
from vespa.package import Schema, Document, Field, FieldSet, RankProfile, HNSW
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import io
import base64
from copy import deepcopy
from textwrap import dedent
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("DeepSearch_Vespa模块初始化")
print(f"当前模块的日志器名称: {logger.name}")

class DeepSearch_Vespa:
    """使用Vespa作为后端的多模态PDF检索系统"""

    def __init__(
            self,
            text_model,
            image_model,
            processor,
            max_iterations: int = 2,
            reranker=None,
            params: dict = None,
            vespa_endpoint: str = None,
            tenant_name: str = None,
            local_deployment: bool = False,
            docker_image: str = "vespaengine/vespa",
            local_port: int = 8080,
            work_dir: str = "./vespa_deepsearch",
            api_key: str = None,
            application_name: str = "pdf-multimodal-search"
    ):
        """
        初始化DeepSearch_Vespa

        Args:
            text_model: 文本嵌入模型
            image_model: 图像嵌入模型
            processor: 图像处理器
            max_iterations: 最大迭代次数
            reranker: 重排序器
            params: 参数字典
            vespa_endpoint: Vespa云端点（如果已有应用）
            tenant_name: Vespa云租户名
            api_key: Vespa云API密钥
            application_name: Vespa应用名称
        """
        self.text_model = text_model
        self.image_model = image_model
        self.processor = processor
        self.max_iterations = max_iterations
        self.reranker = reranker
        self.params = params or {
            "embedding_topk": 15,
            "rerank_topk": 10,
            "text_weight": 0.6,
            "image_weight": 0.4
        }

        self.app = None
        self.app_package = None
        self.tenant_name = tenant_name
        self.api_key = api_key
        self.application_name = application_name
        self.local_deployment = local_deployment
        self.docker_image = docker_image
        self.local_port = local_port
        self.work_dir = work_dir
        self.text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", use_fast=True)
        self.text_embedding_weight = 0.6
        self.embedding_topk = 15

        # 连接到现有Vespa端点或创建新的应用
        if vespa_endpoint:
            logger.info(f"连接到现有Vespa端点: {vespa_endpoint}")
            self.app = Vespa(url=vespa_endpoint)
        elif local_deployment:
            logger.info("准备在本地部署Vespa应用")
            self._create_and_deploy_app_locally()
        elif tenant_name and api_key:
            logger.info(f"使用租户 {tenant_name} 部署Vespa云应用")
            self._create_and_deploy_app()
        else:
            logger.warning("未提供Vespa端点或租户信息，将使用内存检索")

        logger.info("DeepSearch_Vespa初始化完成")

    def _check_if_port_in_use(self, port):
        """检查端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def _check_docker_running(self):
        """检查Docker是否运行中"""
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker未运行或未安装，请确保Docker服务已启动")
            return False

    def _create_and_deploy_app_locally(self):
        """创建并在本地部署Vespa应用"""
        try:
            # 检查Docker是否运行
            if not self._check_docker_running():
                return

            # 检查端口是否被占用
            if self._check_if_port_in_use(self.local_port):
                logger.error(f"端口 {self.local_port} 已被占用，请选择其他端口或关闭占用端口的程序")
                return

            # 创建工作目录
            os.makedirs(self.work_dir, exist_ok=True)
            app_dir = os.path.join(self.work_dir, "application")
            os.makedirs(app_dir, exist_ok=True)

            # 创建容器名称
            container_name = f"vespa-{self.application_name}"

            # 检查是否已存在同名容器
            try:
                result = subprocess.run(
                    ["docker", "ps", "-a", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if container_name in result.stdout.split():
                    # 先停止容器
                    subprocess.run(["docker", "stop", container_name], check=False)
                    # 再移除容器
                    subprocess.run(["docker", "rm", container_name], check=False)
                    logger.info(f"已移除旧的容器: {container_name}")
            except Exception as e:
                logger.warning(f"检查容器时出错: {str(e)}")

            # 创建services.xml文件
            services_path = os.path.join(app_dir, "services.xml")
            with open(services_path, "w") as f:
                f.write(f'''<?xml version="1.0" encoding="utf-8" ?>
                <services version="1.0">
                  <container id="default" version="1.0">
                    <search />
                    <document-api />
                  </container>
                  <content id="pdf_pages" version="1.0">
                    <redundancy>1</redundancy>
                    <documents>
                      <document type="pdf_page" mode="index" />
                    </documents>
                    <nodes>
                      <node distribution-key="0" hostalias="node1" />
                    </nodes>
                  </content>
                </services>
                ''')

            # 创建pdf_page.sd模式定义文件
            schema_path = os.path.join(app_dir, "schemas", "pdf_page.sd")
            os.makedirs(os.path.join(app_dir, "schemas"), exist_ok=True)
            with open(schema_path, "w") as f:
                f.write('''schema pdf_page {
              document pdf_page {
                field id type string {
                  indexing: summary | index
                  match: word
                }
                field pdf_path type string {
                  indexing: summary | index
                }
                field page_index type int {
                  indexing: summary | attribute
                }
                field text type string {
                  indexing: index
                  match: text
                  index: enable-bm25
                }
                # 使用二进制向量表示提高效率
                field embedding type tensor(patch{}, v[16]) {
                  indexing: attribute | index
                  attribute {
                    distance-metric: hamming
                  }
                  index {
                    hnsw {
                      max-links-per-node: 32
                      neighbors-to-explore-at-insert: 400
                    }
                  }
                }
                field text_embedding type tensor<float>(x[384]) {
                  indexing: attribute | index
                  attribute {
                    distance-metric: angular
                  }
                  index {
                    hnsw {
                      max-links-per-node: 16
                      neighbors-to-explore-at-insert: 200
                    }
                  }
                }
              }

              fieldset default {
                fields: text
              }

              rank-profile default {
                inputs {
                  query(qt) tensor<float>(querytoken{}, v[128])
                  query(text_weight) double
                }

                function max_sim() {
                    expression: sum(reduce(sum(query(qt) * unpack_bits(attribute(embedding)), v), max, patch), querytoken)
                }

                function bm25_score() {
                    expression: bm25(text)
                }

                function combinedScore() {
                  expression: query(text_weight) * bm25_score + (1 - query(text_weight)) * max_sim
                }

                first-phase {
                    expression: bm25_score
                }

                second-phase {
                    expression: combinedScore
                    rerank-count: 100
                }
              }
            }
            ''')

            # 创建hosts.xml文件
            hosts_path = os.path.join(app_dir, "hosts.xml")
            with open(hosts_path, "w") as f:
                f.write(f'''<?xml version="1.0" encoding="utf-8" ?>
                <hosts>
                  <host name="localhost">
                    <alias>node1</alias>
                  </host>
                </hosts>
                ''')

            # 使用Docker启动Vespa容器
            mount_path = os.path.abspath(self.work_dir)
            docker_cmd = [
                "docker", "run",
                "-d",
                "--name", container_name,
                "-p", f"{self.local_port}:8080",
                "-p", "19071:19071",
                "-v", f"{mount_path}:/app",
                self.docker_image
            ]

            logger.info(f"启动Docker容器: {' '.join(docker_cmd)}")
            subprocess.run(docker_cmd, check=True)

            # 等待容器启动并服务准备好
            vespa_url = f"http://localhost:{self.local_port}"
            # self._wait_for_vespa_ready(vespa_url)

            # 使用Vespa的deploy命令来部署应用
            deploy_cmd = [
                "docker", "exec", container_name,
                "bash", "-c",
                f"cd /app && /opt/vespa/bin/vespa-deploy prepare /app/application && /opt/vespa/bin/vespa-deploy activate"
            ]

            logger.info(f"部署应用: {' '.join(deploy_cmd)}")
            subprocess.run(deploy_cmd, check=True)

            # 等待应用部署完成
            time.sleep(10)

            # 连接到运行中的Vespa实例
            self.app = Vespa(url=vespa_url)
            logger.info(f"Vespa Docker应用部署成功，端点: {vespa_url}")

        except Exception as e:
            logger.error(f"本地部署Vespa应用失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.app = None

    def _wait_for_vespa_ready(self, url, max_attempts=30, interval=2):
        """等待Vespa服务就绪"""
        logger.info("等待Vespa服务就绪...")

        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{url}/ApplicationStatus")
                if response.status_code == 200:
                    status = response.json()
                    if status.get("status", {}).get("code") == "up":
                        logger.info("Vespa服务已就绪")
                        return True
            except Exception:
                pass

            logger.info(f"Vespa服务尚未就绪，等待中... ({attempt + 1}/{max_attempts})")
            time.sleep(interval)

        logger.warning(f"等待Vespa服务就绪超时，请检查服务是否正常启动")
        return False

    def _compute_text_embedding(self, text: str):
        """计算归一化文本嵌入"""
        if not text or not text.strip():
            return np.zeros(384)

        try:
            with torch.no_grad():
                # 尝试使用不同模型接口
                if hasattr(self.text_model, 'encode'):
                    # BGE/FlagModel方式
                    embedding = self.text_model.encode([text])
                    if isinstance(embedding, torch.Tensor):
                        vector = embedding[0].cpu().numpy()
                    else:
                        vector = embedding[0]
                else:
                    # 其他方式
                    inputs = self.processor.process_queries([text]).to(self.text_model.device)
                    outputs = self.text_model(**inputs)
                    vector = outputs[0].cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs[0]

                # 归一化向量
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                return vector[:384]
        except Exception as e:
            logger.error(f"计算文本嵌入时出错: {str(e)}")
            return np.zeros(384)

    def index_all_pdfs_from_directory(self, directory_path, ocr_method='paddleocr'):
        """
        Index all PDFs from a directory into Vespa

        Args:
            directory_path: Path to the directory containing PDF files
            ocr_method: OCR method to use ('paddleocr' or 'pytesseract')

        Returns:
            bool: True if indexing was successful
        """
        import os
        from pdf2image import convert_from_path
        import json
        from tqdm import tqdm

        logger.info(f"Indexing all PDFs from {directory_path}...")

        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        successfully_index_pdf = []
        pdf_list = [
            '4046173.pdf', '4176503.pdf', '4057524.pdf', '4064501.pdf', '4057121.pdf', '4174854.pdf',
            '4148165.pdf', '4129570.pdf', '4010333.pdf', '4147727.pdf', '4066338.pdf', '4031704.pdf',
            '4050613.pdf', '4072260.pdf', '4091919.pdf', '4094684.pdf', '4063393.pdf', '4132494.pdf',
            '4185438.pdf', '4129670.pdf', '4138347.pdf', '4190947.pdf', '4100212.pdf', '4173940.pdf',
            '4069930.pdf', '4174181.pdf', '4027862.pdf', '4012567.pdf', '4145761.pdf', '4078345.pdf',
            '4061601.pdf', '4170122.pdf', '4077673.pdf', '4107960.pdf', '4005877.pdf', '4196005.pdf',
            '4126467.pdf', '4088173.pdf', '4106951.pdf', '4086173.pdf', '4072232.pdf', '4111230.pdf',
            '4057714.pdf'
        ]
        # Process each PDF file
        for pdf_idx, pdf_file in enumerate(tqdm(pdf_files, desc="Indexing PDFs")):
            if pdf_file not in pdf_list:
                continue
            pdf_path = os.path.join(directory_path, pdf_file)

            try:
                # Convert PDF to images
                pages = convert_from_path(pdf_path)
                logger.info(f"PDF {pdf_file} has {len(pages)} pages")

                # Get OCR text
                ocr_file = os.path.join(directory_path, f"{ocr_method}_save", f"{pdf_file.replace('.pdf', '.json')}")

                if os.path.exists(ocr_file):
                    with open(ocr_file, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                    logger.info(f"Successfully read preprocessed text file: {ocr_file}")
                else:
                    logger.warning(f"OCR file not found: {ocr_file}, skipping PDF")
                    continue

                # Validate page count
                if len(loaded_data) != len(pages):
                    logger.warning(f"OCR data pages ({len(loaded_data)}) do not match PDF pages ({len(pages)})")
                    page_count = min(len(loaded_data), len(pages))
                else:
                    page_count = len(pages)

                # Create documents for each page
                documents = []
                page_keys = list(loaded_data.keys())

                for idx in range(page_count):
                    if idx >= len(pages):
                        break

                    # Check for valid page dimensions
                    page = pages[idx]
                    width, height = page.size
                    if width <= 0 or height <= 0:
                        logger.warning(f"Skipping invalid page {idx + 1}: dimensions {width}x{height}")
                        continue

                    # Get OCR text
                    page_text = loaded_data[page_keys[idx]] if idx < len(page_keys) else ""

                    # Create document structure
                    documents.append({
                        "text": page_text,
                        "image": page,
                        "metadata": {
                            "page_index": idx + 1,
                            "pdf_path": pdf_file
                        }
                    })

                # Index the documents
                if documents:
                    success = self.index_documents(documents)
                    if not success:
                        logger.warning(f"Failed to index documents for {pdf_file}")
                    else:
                        successfully_index_pdf.append(pdf_file)
                else:
                    logger.warning(f"No valid documents created for {pdf_file}")

            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("PDF indexing complete")
        print("PDF indexing complete: ", successfully_index_pdf)
        return True

    def index_documents(self, documents):
        """将文档索引到Vespa"""
        if self.app is None:
            logger.warning("Vespa应用未初始化，无法索引文档")
            return False

        logger.info(f"开始索引 {len(documents)} 个文档...")

        # 创建vespa_feed列表存储所有要索引的文档
        images = [doc["image"] for doc in documents]
        img_embeddings = []
        batch_size = 2

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            with torch.no_grad():
                # 处理图像批次
                batch_inputs = self.processor.process_images(batch_images).to('cpu')
                batch_embeddings = self.image_model(**batch_inputs)
                img_embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

        vespa_feed = []

        for i, (doc, embedding) in enumerate(zip(documents, img_embeddings)):
            # 创建文档ID
            pdf_name = doc["metadata"]["pdf_path"].split('.')[0]
            doc_id = f"{pdf_name}_page{doc['metadata']['page_index']}"
            text = doc.get("text", "")

            # 处理嵌入向量
            embedding_dict = {}
            for idx, patch_embedding in enumerate(embedding):
                # 将向量转换为二进制表示
                binary_vector = (
                    np.packbits(np.where(patch_embedding.numpy() > 0, 1, 0))
                    .astype(np.int8)
                    .tobytes()
                    .hex()
                )
                embedding_dict[idx] = binary_vector

            # 创建Vespa文档
            vespa_doc = {
                "id": doc_id,
                "fields": {
                    "id": doc_id,
                    "pdf_path": doc['metadata']['pdf_path'],
                    "page_index": doc['metadata']['page_index'],
                    "text": doc['text'],
                    "embedding": embedding_dict,
                    # "text_embedding": self._compute_text_embedding(text).tolist()
                }
            }

            vespa_feed.append(vespa_doc)

        # 定义回调函数处理索引结果
        def callback(response, id):
            if not response.is_successful():
                logger.warning(f"Failed to feed document {id}: {response.get_status_code()}")
                logger.warning(response.json)

        # 使用feed_iterable批量索引
        try:
            # 使用较小的批处理大小
            BATCH_SIZE = 2  # 减小批处理大小
            logger.info(f"批量索引 {len(vespa_feed)} 个文档到Vespa...")
            for i in range(0, len(vespa_feed), BATCH_SIZE):
                batch = vespa_feed[i:i + BATCH_SIZE]
                try:
                    logger.info(
                        f"索引批次 {i // BATCH_SIZE + 1}/{(len(vespa_feed) - 1) // BATCH_SIZE + 1}，共 {len(batch)} 个文档...")
                    self.app.feed_iterable(
                        batch,
                        schema="unified",
                        callback=callback,
                        timeout=180  # 增加超时时间
                    )
                    # 每批后短暂暂停
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"批量索引文档时出错 (批次 {i // BATCH_SIZE + 1}): {str(e)}")
                    logger.error(traceback.format_exc())

            logger.info("文档索引完成")
            return True

        except Exception as e:
            logger.error(f"批量索引文档时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _split_query_intent(self, query: str) -> List[str]:
        """将查询拆分为多个不同维度的意图查询"""
        SYSTEM_MESSAGE = dedent("""
        你是一个专业的查询意图分析专家。你的任务是分析用户的查询，并将其拆分为多个不同维度的子查询。

        请遵循以下规则：
        1. 如果查询包含多个不同的信息需求或关注点，请将其拆分为多个子查询
        2. 确保每个子查询关注不同的维度或方面，保证多样性
        3. 不要仅仅改变问题的表述形式，而应该关注不同的信息维度
        4. 如果原始查询已经非常明确且只关注单一维度，则不需要拆分
        5. 子查询应该更加具体和明确，有助于检索到更精准的信息

        请以JSON格式返回，包含以下字段：
        {
            "intent_queries": ["子查询1", "子查询2", ...]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"请分析以下查询并拆分为多个不同维度的子查询：\n\n{query}"}
        ]

        response_format = create_response_format({
            "intent_queries": {
                "type": "array",
                "description": "拆分后的子查询列表",
                "items": {"type": "string"}
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=response_format
        )

        try:
            result = parse_llm_response(response)
            intent_queries = result.get("intent_queries", [query])
            print("intent_queries:", intent_queries)
            return intent_queries if intent_queries else [query]
        except Exception as e:
            logger.error(f"意图拆分出错: {e}")
            return [query]

    def _refine_query_intent(self, original_query: str, intent_queries: List[str], context: str) -> List[str]:
        """基于检索结果细化查询意图"""
        # SYSTEM_MESSAGE = dedent("""
        # 你是一个专业的查询意图优化专家。你的任务是基于已有的检索结果，进一步细化和优化查询意图。

        # 请遵循以下规则：
        # 1. 分析已有检索结果，识别信息缺口和需要进一步探索的方向
        # 2. 基于原始查询和已拆分的意图，生成更加精确的子查询
        # 3. 确保新的子查询能够覆盖原始查询未被满足的信息需求
        # 4. 子查询应该更加具体，包含专业术语和明确的信息需求
        # 5. 避免生成过于相似的子查询，保证多样性

        # 请以JSON格式返回，包含以下字段：
        # {
        #     "refined_intent_queries": ["细化子查询1", "细化子查询2", ...]
        # }
        # """)

        SYSTEM_MESSAGE = dedent("""
        You are a professional query intent optimization expert. Your task is to refine and enhance the user's search intent based on the retrieved content.

        Please follow these guidelines:
        1. Analyze the retrieved content to identify information gaps and areas that require further exploration.
        2. Based on the original query and the decomposed intent queries, generate more precise and targeted sub-queries.
        3. Ensure that the new sub-queries address the information needs that were not fully satisfied by the original query.
        4. Sub-queries should be more specific, incorporating domain-specific terminology and clearly defined information requirements.
        5. Avoid generating overly similar sub-queries; ensure diversity and coverage of different aspects.
        6. Limit the number of refined sub-queries to a maximum of **three**.

        Return your output in JSON format with the following structure:
        {
            "refined_intent_queries": ["Refined sub-query 1", "Refined sub-query 2", ...]
        }
        """)

        # messages = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},
        #     {"role": "user", "content": f"""
        #     原始查询：
        #     {original_query}

        #     已拆分的意图查询：
        #     {json.dumps(intent_queries, ensure_ascii=False)}

        #     已检索到的内容：
        #     {context}

        #     请基于以上信息，细化和优化查询意图：
        #     """}
        # ]

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original query:
            {original_query}

            Decomposed intent queries:
            {json.dumps(intent_queries, ensure_ascii=False)}

            Retrieved context:
            {context}

            Based on the information above, please refine and optimize the search intent:
            """}
        ]

        response_format = create_response_format({
            "refined_intent_queries": {
                "type": "array",
                "description": "细化后的子查询列表",
                "items": {"type": "string"}
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=response_format
        )

        try:
            result = parse_llm_response(response)
            refined_queries = result.get("refined_intent_queries", intent_queries)
            print("Refined intent queries:", refined_queries)
            return refined_queries if refined_queries else intent_queries
        except Exception as e:
            logger.error(f"意图细化出错: {e}")
            return intent_queries

    def _prepare_context(self, search_results: List[Dict[str, str]]) -> str:
        """Prepare context from search results."""
        return '\n'.join([f"{index + 1}. {result['text']}" for index, result in enumerate(search_results)])


    def search_pdf_retrieval(self, data: dict, pdf_path: str):
        """执行搜索检索"""
        if self.app is None:
            raise ValueError("Vespa应用未初始化")
        original_query = deepcopy(data['query'])
        data_ori = deepcopy(data)

        # 2. 拆分查询意图
        intent_queries = self._split_query_intent(original_query)
        logger.info(f"🔍 意图拆分结果: {intent_queries}")

        # 3. 对每个意图进行检索
        all_results = []
        seen_texts = set()

        pdf_filename = os.path.basename(pdf_path)
        print("pdf_filename: ", pdf_filename)

        for intent_idx, intent_query in enumerate(intent_queries):
            logger.info(f"检索意图 {intent_idx + 1}/{len(intent_queries)}: {intent_query}")

            # 使用Vespa检索
            # results = self.retrieve(intent_query, data['documents'])
            results = self._search_with_vespa(intent_query, data['documents'][0]['metadata']['pdf_path'])
            results = results[:self.embedding_topk // len(intent_queries)]

            # 合并结果，去重
            for result in results:
                if result['text'] not in seen_texts:
                    seen_texts.add(result['text'])
                    all_results.append(result)

        # # 第三步：基于第一轮检索结果进行意图细化
        # refined_intent_queries = self._refine_query_intent(original_query, intent_queries,
        #                                                    json.dumps(all_results, ensure_ascii=False,
        #                                                               indent=2))
        # logger.info(f"🔍 意图细化结果: {refined_intent_queries}")
        #
        # for refine_idx, refine_query in enumerate(refined_intent_queries):
        #     logger.info(f"检索意图 {refine_idx + 1}/{len(refined_intent_queries)}: {refine_query}")
        #
        #     # 使用Vespa检索
        #     # results = self.retrieve(refine_query, data_ori['documents'])
        #     results = self._search_with_vespa(refine_query, len(data['documents']),
        #                                       data['documents'][0]['metadata']['pdf_path'])
        #
        #     # 合并结果，去重
        #     for result in results:
        #         if result['text'] not in seen_texts:
        #             seen_texts.add(result['text'])
        #             all_results.append(result)

        # # 4. 使用reranker重排序
        # if self.reranker and all_results:
        #     final_search_results = self._rerank_results(original_query, all_results)
        #     logger.info(f"📊 最终结果: {len(final_search_results)} 条")
        #     logger.info([doc['score'] for doc in final_search_results])
        #     return final_search_results

        # 按分数排序
        logger.info([doc['score'] for doc in sorted(all_results, key=lambda x: x['score'], reverse=True)[:self.params['rerank_topk']]])
        return sorted(all_results, key=lambda x: x['score'], reverse=True)[:self.params['rerank_topk']]

    def retrieve(self, query: str, documents: List[dict], save_path: Optional[str] = None) -> List[dict]:
        results = self._search_with_vespa(query, documents[0]['metadata']['pdf_path'])
        for doc in documents:
            try:
                # 获取文本和图像
                matching_scores = [x['score'] for x in results if x["page"] == doc['metadata']['page_index']]
                if len(matching_scores) == 0:
                    continue
                text = doc.get("text", "")
                text_score = self._compute_text_score(query, text)

                image_score = matching_scores[0]
                combined_score = self._combine_scores(text_score, image_score)

                doc["score"] = combined_score
                doc["metadata"] = doc.get("metadata", {})  # Ensure metadata exists
            except Exception as e:
                # 处理任何其他错误
                print(f"处理文档时出错: {str(e)}")
                traceback.print_exc()
                continue
        for doc in documents:
            if "metadata" in doc and "page_index" not in doc["metadata"]:
                doc["metadata"]["page_index"] = None  # Default to None if page_index is missing


        return sorted(documents, key=lambda x: x["score"], reverse=True)[:self.params['rerank_topk']]

    def _search_with_vespa(self, query, pdf_filter=None):
        """使用Vespa执行搜索"""
        try:
            # 计算查询嵌入
            logger.info("利用vespa加速colpali中...")
            query_inputs = self.processor.process_queries([query]).to('cpu')
            with torch.no_grad():
                query_embeddings = self.image_model(**query_inputs)
            query_embeddings = query_embeddings.to("cpu")
            query_embedding = torch.unbind(query_embeddings)[0]

            # 处理浮点查询嵌入
            float_query_embedding = {str(k): v.tolist() for k, v in enumerate(query_embedding)}

            # 处理二进制查询嵌入
            binary_query_embeddings = {}
            for k, v in float_query_embedding.items():
                binary_query_embeddings[k] = (
                    np.packbits(np.where(np.array(v) > 0, 1, 0))
                    .astype(np.int8)
                    .tobytes()
                    .hex()
                )

            # 构建查询张量
            query_tensors = {
                "input.query(qtb)": binary_query_embeddings,
                "input.query(qt)": float_query_embedding,
            }
            for i in range(len(binary_query_embeddings)):
                query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[str(i)]

            # 构建最近邻查询部分
            target_hits_per_query_tensor = 20  # 可调整的超参数
            nn = []
            for i in range(len(binary_query_embeddings)):
                nn.append(
                    f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
                )
            # 使用OR运算符组合最近邻运算符
            nn_expr = " OR ".join(nn)

            # 构建YQL查询
            yql_query = f"select * from unified where ({nn_expr})"
            if pdf_filter:
                yql_query += f" and pdf_path contains '{pdf_filter}'"

            response = self.app.query(
                yql=yql_query,
                ranking="retrieval-and-rerank",
                timeout=120,
                hits=15,
                body={**query_tensors, "presentation.timing": True},
            )

            # vespa_qt_format = {}
            # if hasattr(query_embedding, 'shape') and len(query_embedding.shape) > 1:
            #     # 多patch/token格式 - 例如 (n_patches, vector_dim)
            #     for i, patch_embedding in enumerate(query_embedding):
            #         # 将向量转换为列表
            #         vector_values = patch_embedding.tolist()
            #         # 添加到vespa格式
            #         vespa_qt_format[i] = vector_values
            #
            # else:
            #     # 单一向量格式
            #     vector_values = query_embedding.tolist()
            #     vespa_qt_format[0] = vector_values
            #
            # request_body = {
            #     "input.query(qt)": vespa_qt_format,
            #     "input.query(text_weight)": 0.6,  # 可以根据需要调整文本权重
            #     "presentation.timing": True
            # }
            #
            # response = self.app.query(
            #     yql=f"select * from pdf_page where userInput(@userQuery) and pdf_path contains '{pdf_filter}'",
            #     ranking="retrieval-and-rerank",
            #     userQuery=query,
            #     timeout=120,
            #     hits=hits,  # 返回前5个结果
            #     body=request_body,
            # )

            # 调试信息
            logger.info(f"查询响应状态: {response.status_code if hasattr(response, 'status_code') else '无状态码'}")

            def get_text_by_docapi(doc_id, vespa_url="http://localhost:8080"):
                url = f"{vespa_url}/document/v1/unified/unified/docid/{doc_id}"
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    fields = data.get("fields", {})
                    text = fields.get("text", "")
                    return text
                else:
                    print(f"请求失败: {response.status_code}")
                    return None

            # 处理结果
            if response.is_successful():
                logger.info(f"查询成功，返回 {len(response.hits)} 条结果")
                results = []
                for hit in response.hits:
                    fields = hit['fields']
                    doc_id = fields['id']
                    text = get_text_by_docapi(doc_id=doc_id)
                    results.append({
                        "score": hit['relevance'],
                        "page": fields['page_index'],
                        "text": text,
                        "metadata": {
                            "page_index": fields['page_index'],
                            "pdf_path": fields['pdf_path']
                        }
                    })
                print("results: ", results)
                return sorted(results, key=lambda x: x["score"], reverse=True)[:15]
            else:
                error_msg = response.json if hasattr(response, 'json') else "未知错误"
                logger.warning(f"Vespa查询失败: {error_msg}")
                return []

        except Exception as e:
            logger.error(f"Vespa查询出错: {str(e)}")
            logger.error(traceback.format_exc())  # 打印完整堆栈信息
            return []

    def _combine_scores(self, text_score: float, image_score: float) -> float:
        image_score = image_score / 100
        # total_weight = self.embedding_weight + (1 - self.embedding_weight)
        text_weight = self.params['text_weight']
        image_weight = (1 - text_weight)
        return text_weight * text_score + image_weight * image_score

    def _compute_text_score(self, query: str, text: str) -> float:
        bge_model_name = "BAAI/bge-large-en-v1.5"
        text_tokenizer = AutoTokenizer.from_pretrained(bge_model_name, use_fast=True)
        text_model = AutoModel.from_pretrained(bge_model_name).to("cpu")
        if not text.strip():
            return 0.0
        query_tokens = text_tokenizer(query, padding=True, truncation=True, return_tensors="pt",
                                           max_length=512).to("cpu")
        text_tokens = text_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cpu")
        with torch.no_grad():
            query_embedding = text_model(**query_tokens)
            text_embedding = text_model(**text_tokens)
        query_embedding = query_embedding.last_hidden_state[:, 0].cpu().numpy()
        text_embedding = text_embedding.last_hidden_state[:, 0].cpu().numpy()
        # 归一化
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        return float(query_embedding @ text_embedding.T)

    def _rerank_results(self, query, results):
        """使用reranker重新排序结果"""
        try:
            pairs = [[query, result["text"]] for result in results]
            rerank_scores = self.reranker.compute_score(pairs, normalize=True)

            # 更新分数
            for i, score in enumerate(rerank_scores):
                results[i]["score"] = float(score)

            # 排序
            return sorted(results, key=lambda x: x["score"], reverse=True)[:self.params['rerank_topk']]
        except Exception as e:
            logger.error(f"重排序出错: {str(e)}")
            return results

def parse_llm_response(response_text: str) -> dict:
    """
    从LLM响应中提取JSON数据，处理各种可能的格式

    Args:
        response_text: 模型返回的原始文本

    Returns:
        dict: 解析后的JSON对象
    """
    import re
    import json

    # 1. 清理可能的markdown代码块格式
    cleaned_text = re.sub(r'```(?:json|python)?', '', response_text)
    cleaned_text = re.sub(r'`', '', cleaned_text).strip()

    # 2. 尝试直接解析JSON
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # 3. 尝试查找JSON内容
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, cleaned_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # 4. 回退方案：手动提取关键字段
    output_dict = {}

    # 提取refined_intent_queries数组
    queries_pattern = r'"refined_intent_queries"\s*:\s*\[(.*?)\]'
    queries_match = re.search(queries_pattern, cleaned_text, re.DOTALL)
    if queries_match:
        query_items = re.findall(r'"([^"]+)"', queries_match.group(1))
        output_dict["refined_intent_queries"] = query_items

    # 提取intent_queries数组（如果有）
    intent_pattern = r'"intent_queries"\s*:\s*\[(.*?)\]'
    intent_match = re.search(intent_pattern, cleaned_text, re.DOTALL)
    if intent_match:
        intent_items = re.findall(r'"([^"]+)"', intent_match.group(1))
        output_dict["intent_queries"] = intent_items

    return output_dict


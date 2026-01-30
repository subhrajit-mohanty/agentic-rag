"""
Kubernetes Code Executor

Creates and manages Kubernetes Jobs for safe code execution.
This is used by the CodeExecutorTool when running in Kubernetes mode.
"""

import asyncio
import base64
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution."""
    namespace: str = "agentic-rag"
    image: str = "python:3.11-slim"
    memory_limit: str = "256Mi"
    cpu_limit: str = "500m"
    timeout_seconds: int = 30
    service_account: str = "tool-executor"
    enable_network: bool = False
    ttl_seconds_after_finished: int = 60


class KubernetesCodeExecutor:
    """
    Executes code safely in Kubernetes Jobs.
    
    Features:
    - Isolated execution environment
    - Resource limits
    - Network isolation (optional)
    - Automatic cleanup
    - Timeout handling
    """
    
    def __init__(
        self,
        config: Optional[CodeExecutionConfig] = None,
        k8s_client: Any = None
    ):
        self.config = config or CodeExecutionConfig()
        self._client = k8s_client
        self._initialized = False
    
    async def _initialize(self) -> None:
        """Initialize Kubernetes client."""
        if self._initialized:
            return
        
        try:
            from kubernetes import client, config as k8s_config
            from kubernetes.client import ApiException
            
            # Try in-cluster config first, then local
            try:
                k8s_config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
                logger.info("Loaded local Kubernetes config")
            
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
            self._initialized = True
            
        except ImportError:
            logger.error("kubernetes package not installed. Install with: pip install kubernetes")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    async def execute(
        self,
        code: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute code in a Kubernetes Job.
        
        Args:
            code: Python code to execute
            inputs: Input variables to inject
            timeout: Execution timeout (overrides config)
            
        Returns:
            Execution result with stdout, stderr, exit_code
        """
        await self._initialize()
        
        job_name = self._generate_job_name()
        timeout = timeout or self.config.timeout_seconds
        
        try:
            # Create Job
            job = await self._create_job(job_name, code, inputs)
            
            # Wait for completion
            result = await self._wait_for_completion(job_name, timeout)
            
            return result
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "success": False,
                "error": str(e)
            }
            
        finally:
            # Cleanup
            await self._delete_job(job_name)
    
    def _generate_job_name(self) -> str:
        """Generate unique job name."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(
            f"{timestamp}{id(self)}".encode()
        ).hexdigest()[:6]
        return f"code-exec-{timestamp}-{random_suffix}"
    
    async def _create_job(
        self,
        job_name: str,
        code: str,
        inputs: Optional[Dict[str, Any]]
    ) -> Any:
        """Create Kubernetes Job."""
        from kubernetes import client
        
        # Prepare code with inputs
        full_code = self._prepare_code(code, inputs)
        
        # Base64 encode the code for safe transmission
        code_b64 = base64.b64encode(full_code.encode()).decode()
        
        # Job spec
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.config.namespace,
                labels={
                    "app": "code-executor",
                    "job-name": job_name
                }
            ),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=self.config.ttl_seconds_after_finished,
                backoff_limit=0,  # No retries
                active_deadline_seconds=self.config.timeout_seconds,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "code-executor",
                            "job-name": job_name
                        }
                    ),
                    spec=client.V1PodSpec(
                        service_account_name=self.config.service_account,
                        restart_policy="Never",
                        containers=[
                            client.V1Container(
                                name="executor",
                                image=self.config.image,
                                command=[
                                    "sh", "-c",
                                    f"echo {code_b64} | base64 -d | python"
                                ],
                                resources=client.V1ResourceRequirements(
                                    limits={
                                        "memory": self.config.memory_limit,
                                        "cpu": self.config.cpu_limit
                                    },
                                    requests={
                                        "memory": "64Mi",
                                        "cpu": "100m"
                                    }
                                ),
                                security_context=client.V1SecurityContext(
                                    read_only_root_filesystem=True,
                                    run_as_non_root=True,
                                    run_as_user=1000,
                                    allow_privilege_escalation=False,
                                    capabilities=client.V1Capabilities(
                                        drop=["ALL"]
                                    )
                                ),
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="tmp",
                                        mount_path="/tmp"
                                    )
                                ]
                            )
                        ],
                        volumes=[
                            client.V1Volume(
                                name="tmp",
                                empty_dir=client.V1EmptyDirVolumeSource(
                                    size_limit="10Mi"
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        # Create the job
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._batch_api.create_namespaced_job(
                namespace=self.config.namespace,
                body=job
            )
        )
        
        logger.info(f"Created job: {job_name}")
        return result
    
    async def _wait_for_completion(
        self,
        job_name: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Wait for job completion and get results."""
        from kubernetes.client import ApiException
        
        start_time = datetime.utcnow()
        
        while True:
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                return {
                    "stdout": "",
                    "stderr": "Execution timeout",
                    "exit_code": 124,
                    "success": False,
                    "error": "timeout"
                }
            
            # Get job status
            loop = asyncio.get_event_loop()
            try:
                job = await loop.run_in_executor(
                    None,
                    lambda: self._batch_api.read_namespaced_job_status(
                        name=job_name,
                        namespace=self.config.namespace
                    )
                )
            except ApiException as e:
                if e.status == 404:
                    return {
                        "stdout": "",
                        "stderr": "Job not found",
                        "exit_code": 1,
                        "success": False,
                        "error": "job_not_found"
                    }
                raise
            
            # Check if completed
            if job.status.succeeded is not None and job.status.succeeded > 0:
                # Get pod logs
                logs = await self._get_pod_logs(job_name)
                return {
                    "stdout": logs,
                    "stderr": "",
                    "exit_code": 0,
                    "success": True
                }
            
            if job.status.failed is not None and job.status.failed > 0:
                # Get pod logs for error
                logs = await self._get_pod_logs(job_name)
                return {
                    "stdout": "",
                    "stderr": logs,
                    "exit_code": 1,
                    "success": False,
                    "error": "execution_failed"
                }
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    async def _get_pod_logs(self, job_name: str) -> str:
        """Get logs from the job's pod."""
        loop = asyncio.get_event_loop()
        
        try:
            # Find the pod
            pods = await loop.run_in_executor(
                None,
                lambda: self._core_api.list_namespaced_pod(
                    namespace=self.config.namespace,
                    label_selector=f"job-name={job_name}"
                )
            )
            
            if not pods.items:
                return "No pod found"
            
            pod_name = pods.items[0].metadata.name
            
            # Get logs
            logs = await loop.run_in_executor(
                None,
                lambda: self._core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.config.namespace,
                    container="executor"
                )
            )
            
            return logs[:10000]  # Limit log size
            
        except Exception as e:
            logger.error(f"Failed to get pod logs: {e}")
            return f"Failed to get logs: {e}"
    
    async def _delete_job(self, job_name: str) -> None:
        """Delete the job and its pods."""
        from kubernetes import client
        from kubernetes.client import ApiException
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._batch_api.delete_namespaced_job(
                    name=job_name,
                    namespace=self.config.namespace,
                    body=client.V1DeleteOptions(
                        propagation_policy="Background"
                    )
                )
            )
            logger.debug(f"Deleted job: {job_name}")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete job {job_name}: {e}")
    
    def _prepare_code(
        self,
        code: str,
        inputs: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare code with input injection."""
        if not inputs:
            return code
        
        # Inject inputs at the beginning
        input_lines = []
        for key, value in inputs.items():
            if isinstance(value, str):
                input_lines.append(f'{key} = """{value}"""')
            else:
                input_lines.append(f"{key} = {repr(value)}")
        
        return "\n".join(input_lines) + "\n\n" + code


# Job template for reference (can be applied manually)
JOB_TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata:
  name: code-exec-{job_id}
  namespace: agentic-rag
  labels:
    app: code-executor
spec:
  ttlSecondsAfterFinished: 60
  backoffLimit: 0
  activeDeadlineSeconds: 30
  template:
    metadata:
      labels:
        app: code-executor
    spec:
      serviceAccountName: tool-executor
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: executor
        image: python:3.11-slim
        command: ["python", "-c"]
        args:
        - |
          {code}
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
          requests:
            memory: "64Mi"
            cpu: "100m"
        securityContext:
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 10Mi
"""

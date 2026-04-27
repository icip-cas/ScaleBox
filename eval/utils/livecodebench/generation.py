import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
import os

from datasets import load_dataset
from tqdm import tqdm


LCB_ALLOWED_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

_LCB_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6"]
for version in _LCB_VERSIONS:
    LCB_ALLOWED_FILES[version] = [f"test{version[1:]}.jsonl" if version != "v1" else "test.jsonl"]

for start_index in range(1, len(_LCB_VERSIONS) + 1):
    for end_index in range(start_index + 1, len(_LCB_VERSIONS) + 1):
        start_version = _LCB_VERSIONS[start_index - 1]
        end_version = _LCB_VERSIONS[end_index - 1]
        LCB_ALLOWED_FILES[f"{start_version}_{end_version}"] = [
            f"test{index}.jsonl" if index != 1 else "test.jsonl"
            for index in range(start_index, end_index + 1)
        ]


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        if self.metadata.get("func_name", None) == None:
            res = {
                    "type": "stdin_stdout",
                    "input": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "output": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": None,
            }
        else:
            res = {
                "type": "function_call",
                "input": [
                    [json.loads(line) for line in t.input.split("\n")]
                    for t in self.public_test_cases + self.private_test_cases
                ],
                "output": [
                    [json.loads(line) for line in t.output.split("\n")]
                    for t in self.public_test_cases + self.private_test_cases
                ],
                "fn_name": self.metadata.get("func_name", None),
            }
        return res


def _resolve_lcb_dataset_dir(dataset_path: str) -> str:
    normalized_path = os.path.abspath(dataset_path)
    if os.path.basename(normalized_path) == "code_generation_lite":
        return normalized_path

    nested_path = os.path.join(normalized_path, "code_generation_lite")
    if os.path.isdir(nested_path):
        return nested_path

    return normalized_path


def _get_lcb_files_for_version(release_version: str) -> list[str]:
    if release_version not in LCB_ALLOWED_FILES:
        raise ValueError(f"Unsupported livecodebench version: {release_version}")
    return LCB_ALLOWED_FILES[release_version]


def load_local_code_generation_dataset(
    dataset_path: str,
    release_version="release_v1",
    start_date=None,
    end_date=None,
) -> list[CodeGenerationProblem]:
    dataset_dir = _resolve_lcb_dataset_dir(dataset_path)
    file_names = _get_lcb_files_for_version(release_version)

    raw_samples = []
    for file_name in tqdm(file_names, desc="Loading LiveCodeBench files", ncols=120):
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"LiveCodeBench file does not exist: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))

    dataset = [CodeGenerationProblem(**sample) for sample in raw_samples]
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [sample for sample in dataset if p_start_date <= sample.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [sample for sample in dataset if sample.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_cpp_generation_dataset(
    release_version="release_v1", start_date=None, end_date=None
) -> list[CodeGenerationProblem]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "..", "livecodebench-cpp", "code_generation_lite")
    dataset = load_dataset(
        dataset_path,
        split="test",
        version_tag=release_version,
        trust_remote_code=True,
    )
    dataset = [CodeGenerationProblem(**p) for p in tqdm(dataset, desc="Building CodeGenerationProblem")]
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


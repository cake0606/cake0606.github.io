## Sequence Group
### decoding 策略
- Parallel Sampling： 基于一个prompt， 给出n种不同的output
- Beam Search： 每个推理阶段产出top k个output， k被称为Beam width
在vllm设计中， 采用fan-in和fan-out
### Parallel Sampling
#### ParentRequest
`ParentRequest`只负责管理
- 保存父请求元信息
	- 父请求id`request_id`
	- 对外的id`external_req_id`
	- 原始`sampling_params`
- 生成child request的信息
	- 把父请求参数改造成child req参数
- 维护父子关系
	- 记录哪些child id还未结束
- 聚合多个child的输出
	- 流式模式下直接返回child输出
	- `FINAL_ONLY`模式下， 等所有child完成， 再一次性聚合返回
- 父请求级别的统计归并

#### 流程
1. 请求进入时：先创建`ParentRequest`, 再生成子请求元信息
2. prefill阶段：`ParentRequest`不参与， Scheduler只能看到多个child request
3. decode阶段： 每个child request独立decode。直到触发`make_request_output()`， 如果有parent_req, 就会调用`parent_req.get_outputs(self.request_id, output)`。
4. 所有child完成后， 清理掉parent
流式模式下输出类似于：
`text
RequestOutput(request_id="reqA", outputs=[CompletionOutput(index=1, text="Hello")], finished=False)
RequestOutput(request_id="reqA", outputs=[CompletionOutput(index=0, text="Hi")], finished=False)
RequestOutput(request_id="reqA", outputs=[CompletionOutput(index=2, text="Hey")], finished=False)
...
RequestOutput(request_id="reqA", outputs=[CompletionOutput(index=1, text="Hello world")], finished=False)
...
RequestOutput(request_id="reqA", outputs=[CompletionOutput(index=2, text="Hey there")], finished=True)  # 最后一批时可能 finished=True
`
`FINAL_ONLY`模式下输出类似于：
`
RequestOutput(
    request_id="reqA",
    outputs=[
        CompletionOutput(index=0, text="..."),
        CompletionOutput(index=1, text="..."),
        CompletionOutput(index=2, text="..."),
    ],
    finished=True,
)
`
### Beam Search
#### BeamSearchInstance
作为一个prompt的beam search上下文管理器， 管理BeamSearchSequence和完成列表
#### 流程
1. 把prompt变成`BeamSearchInstance`
2. 每一轮只让当前活跃beam向前走1个token
3. 在`_beam_search_step`中取出所有活跃beam
4. 记录每个prompt在all_beams中的切片范围
5. 对每条活跃beam发起一次一步长请求
6. 每条beam拓展出若干新候选
7. 拓展完马上裁剪， 只保留beamwidth个候选
8. 从已完成和活跃的beam（搜索完成时还未结束、不能继续拓展）中选择最好的一批











/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/auto_schedule.cc
 * \brief The user interface and tuning options of the TVM auto-scheduler.
 */

#include <tvm/auto_scheduler/auto_schedule.h>
#include <tvm/runtime/registry.h>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(TuningOptionsNode);

TuningOptions::TuningOptions(int num_measure_trials, int early_stopping, int num_measures_per_round,
                             int verbose, ProgramBuilder builder, ProgramRunner runner,
                             Optional<Array<MeasureCallback>> measure_callbacks) {
  auto node = make_object<TuningOptionsNode>();
  node->num_measure_trials = num_measure_trials;
  node->early_stopping = early_stopping;
  node->num_measures_per_round = num_measures_per_round;
  node->verbose = verbose;
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->measure_callbacks = std::move(measure_callbacks);
  data_ = std::move(node);
}

std::pair<te::Schedule, Array<te::Tensor>> AutoSchedule(SearchPolicy search_policy,
                                                        TuningOptions tuning_options) {
  // Create a ProgramMeasurer to handle the schedule build and performance measure
  ProgramMeasurer measurer =
      ProgramMeasurer(tuning_options->builder, tuning_options->runner,
                      tuning_options->measure_callbacks, tuning_options->verbose);
  // Search for the best schedule
  State state =
      search_policy->Search(tuning_options->num_measure_trials, tuning_options->early_stopping,
                            tuning_options->num_measures_per_round, measurer);
  if (state.defined()) {
    return search_policy->search_task->compute_dag.ApplySteps(state->transform_steps);
  } else {
    StdCout(tuning_options->verbose)
        << "No valid state found in this search round. Check if it has traversed all of the "
        << "search space." << std::endl;
    // Return the default schedule
    return {te::Schedule(search_policy->search_task->compute_dag->ops),
            search_policy->search_task->compute_dag->tensors};
  }
}



//my helper founction
ProgramMeasurer MyMeasureStoreBestState(SearchPolicy search_policy, TuningOptions tuning_options,
	const Array<State>& tuned_states){//, int& tot_config_num, double& best_result) {
	Array<MeasureInput> inputs;
	Array<MeasureResult> results;
	for (State state : tuned_states) {
		inputs.push_back(MeasureInput(search_policy->search_task, state));
	}
	// Create a ProgramMeasurer to handle the schedule build and performance measure
	ProgramMeasurer measurer =
		ProgramMeasurer(tuning_options->builder, tuning_options->runner,
						tuning_options->measure_callbacks, tuning_options->verbose);
	StdCout(search_policy->verbose) << Chars('-', 70) << "\n"
		<< Chars('-', 30) << "  [ " << "Measure" << " ]\n"
		<< Chars('-', 70) << std::endl;
	//print("Measure", search_policy->verbose);
	results = measurer->Measure(search_policy->search_task, search_policy, inputs);
	//int tot_config_num = inputs.size(); //update the number of configs being measured
	StdCout(search_policy->verbose) << Chars('-', 70) << "\n"
		<< Chars('-', 30) << "  [ " << "Done" << " ]\n"
		<< Chars('-', 70) << std::endl;
	//PrintTitle("Done", search_policy->verbose);
	return measurer;
}


State MyTestWhyNotMeasure(SearchPolicy search_policy, TuningOptions tuning_options) {
	// Create a ProgramMeasurer to handle the schedule build and performance measure
	ProgramMeasurer measurer =
		ProgramMeasurer(tuning_options->builder, tuning_options->runner,
			tuning_options->measure_callbacks, tuning_options->verbose);
	// Search for the best schedule
	State state =
		search_policy->Search(tuning_options->num_measure_trials, tuning_options->early_stopping,
			tuning_options->num_measures_per_round, measurer);
	
	if (state.defined()) {
		//return search_policy->search_task->compute_dag.ApplySteps(state->transform_steps);
		//try to measure it using the measurer
		Array<MeasureInput> inputs;
		Array<MeasureResult> results;
		StdCout(search_policy->verbose) << Chars('-', 70) << "\n"
			<< Chars('-', 30) << "  [ " << "Measure" << " ]\n"
			<< Chars('-', 70) << std::endl;
		inputs.push_back(MeasureInput(search_policy->search_task, state));
		results = measurer->Measure(search_policy->search_task, search_policy, inputs);
		StdCout(search_policy->verbose) << 1.0 / FloatArrayMean(results[0]->costs) << (measurer->best_flops[search_policy->search_task->workload_key]) / (search_policy->search_task->compute_dag->flop_ct) << std::endl;
		StdCout(search_policy->verbose) << Chars('-', 70) << "\n"
			<< Chars('-', 30) << "  [ " << "Done" << " ]\n"
			<< Chars('-', 70) << std::endl;
	}
	else {
		StdCout(tuning_options->verbose)
			<< "No valid state found in this search round. Check if it has traversed all of the "
			<< "search space." << std::endl;
		// Return the default schedule
		/*return { te::Schedule(search_policy->search_task->compute_dag->ops),
			search_policy->search_task->compute_dag->tensors };*/
	}
	return state;
}


Array<State> MyGetStatesFromTunedKnobs11(//SearchPolicy search_policy, //TuningOptions tuning_options,
	const ComputeDAG& dag_to_tune, const State& state_reused_from,
	const Array<Array<Array<Integer>>>& tile_sizes,
	const Array<Integer>& multi_split_step_ids, const Array<Integer>& vector_split_step_ids) {
	//Array<Optional<Integer>>& tot_config_num, Array<double>& best_result
	//) {
	// given tile_sizes and the split step ids for multi tiling of thread binding, generate states for measure, and return the best one from them.
	const Array<Step>& transform_steps = state_reused_from->transform_steps;
	//const State& init_state = dag_to_tune->init_state;
	size_t config_num = tile_sizes.size();

	//std::cout << "TOT NUM OF configs: " << config_num << std::endl;

	Array<State> tuned_states;
	for (size_t config_i = 0; config_i < config_num; ++config_i) {

		//std::cout << "which config: " << config_i << std::endl;

		//build state for each config
		State tmp_s = dag_to_tune->init_state;
		size_t split_step_i = 0;
		size_t vector_split_step_i = 0;
		for (int i = 0; i < (int)transform_steps.size(); i++) {

			//std::cout << "which step: " << i << std::endl;

			if ((split_step_i < multi_split_step_ids.size()) && (i == GetIntImm(multi_split_step_ids[split_step_i]))) {

				//std::cout << "which multi split step: " << split_step_i << "  the order in seq: " << GetIntImm(multi_split_step_ids[split_step_i]) << std::endl;

				const Step& step_reuse = transform_steps[i];
				auto ps = step_reuse.as<SplitStepNode>();
				//this is one split step that we have tuned
				SplitStep step = SplitStep(ps->stage_id, ps->iter_id, ps->extent,
					Array<Optional<Integer>>(tile_sizes[config_i][split_step_i].begin(), tile_sizes[config_i][split_step_i].end()), ps->inner_to_outer);
				split_step_i++;
				tmp_s.CopyOnWrite()->transform_steps.push_back(step);
				StepApplyToState(step, &tmp_s, dag_to_tune);
			}
			else if ((vector_split_step_i < vector_split_step_ids.size()) && (i == GetIntImm(vector_split_step_ids[vector_split_step_i]))) {

				//std::cout << "which vector split step: " << vector_split_step_i << "  the order in seq: " << GetIntImm(vector_split_step_ids[vector_split_step_i]) << std::endl;

				const Step& step_reuse = transform_steps[i];
				auto ps = step_reuse.as<SplitStepNode>();
				//this is one split step that for the vectorization in cooperative fetching; the default vectorization value is 1.
				SplitStep step = SplitStep(ps->stage_id, ps->iter_id, ps->extent,
					Array<Optional<Integer>>(tile_sizes[config_i][multi_split_step_ids.size() + vector_split_step_i].begin(), tile_sizes[config_i][multi_split_step_ids.size() + vector_split_step_i].end()),
					ps->inner_to_outer);
				vector_split_step_i++;
				tmp_s.CopyOnWrite()->transform_steps.push_back(step);
				StepApplyToState(step, &tmp_s, dag_to_tune);
			}
			else {

				//std::cout << "NOT A SPLIT SEQ " << std::endl;

				//directly copy the step from state_reused_from
				const Step& step = transform_steps[i];
				tmp_s.CopyOnWrite()->transform_steps.push_back(step);
				StepApplyToState(step, &tmp_s, dag_to_tune);
			}
		}
		//we get a state with transform steps applied
		tuned_states.push_back(dag_to_tune.InferBound(tmp_s));
	}

	//std::cout << "BEFORE WE InferBound" << std::endl;
	//std::cout << "TOT NUM OF states: " << tuned_states.size() << std::endl;

	return tuned_states;
	//measure the states we generated
	//tuned_states = dag_to_tune.InferBound(tuned_states);
	//return tuned_states;
	/*Array<MeasureInput> inputs;
	Array<MeasureResult> results;
	for (State state : tuned_states) {
	inputs.push_back(MeasureInput(search_policy->search_task, state));
	}
	// Create a ProgramMeasurer to handle the schedule build and performance measure
	ProgramMeasurer measurer =
	ProgramMeasurer(tuning_options->builder, tuning_options->runner,
	tuning_options->measure_callbacks, tuning_options->verbose);
	PrintTitle("Measure", search_policy->verbose);
	results = measurer->Measure(search_policy->search_task, search_policy, inputs);
	tot_config_num[0] = tot_config_num[0] + inputs.size(); //update the number of configs being measured
	PrintTitle("Done", search_policy->verbose);

	Array<State> ret_state_array;
	State ret_state =  measurer->best_state[search_policy->search_task->workload_key];
	if (ret_state.defined()) {
	ret_state_array.push_back(ret_state);
	best_result.push_back(measurer->best_flops[search_policy->search_task->workload_key]);
	}
	else {
	StdCout(tuning_options->verbose)
	<< "No valid state found in this search round. Check if it has traversed all of the "
	<< "search space." << std::endl;
	// Return empty array
	}
	return ret_state_array;*/
}




TVM_REGISTER_GLOBAL("auto_scheduler.TuningOptions")
    .set_body_typed([](int num_measure_trials, int early_stopping, int num_measures_per_round,
                       int verbose, ProgramBuilder builder, ProgramRunner runner,
                       Optional<Array<MeasureCallback>> measure_callbacks) {
      return TuningOptions(num_measure_trials, early_stopping, num_measures_per_round, verbose,
                           builder, runner, measure_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.AutoSchedule")
    .set_body_typed([](SearchPolicy search_policy, TuningOptions tuning_options) {
      te::Schedule sch;
      Array<te::Tensor> return_tensors;
      std::tie(sch, return_tensors) = AutoSchedule(search_policy, tuning_options);
      return Array<ObjectRef>{sch, return_tensors};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.MyMeasureStoreBestState")
	.set_body_typed(MyMeasureStoreBestState);


TVM_REGISTER_GLOBAL("auto_scheduler.MyTestWhyNotMeasure")
	.set_body_typed(MyTestWhyNotMeasure);

TVM_REGISTER_GLOBAL("auto_scheduler.MyGetStatesFromTunedKnobs11")
	.set_body_typed(MyGetStatesFromTunedKnobs11);

}  // namespace auto_scheduler
}  // namespace tvm

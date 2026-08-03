#!/bin/bash

# -----------------------------------------------------------------
# PARALLEL PATTERN
# NODE-WISE PARALLELISM ; SHELL INTERFACING EXAMPLE
# -----------------------------------------------------------------

export MAX_PARALLEL=6
export CORES_PER_JOB=6
export REQUEST_DATE="20260731"

run_parallel() {

    local worklist="$1"
    local update_time
    update_time=$(date +"%Y-%m-%d")

    # -----------------------------------
    # CREATE PID SLOT : RESOURCE MANAGER
    # -----------------------------------
    local free_slots=()
    local slot
    # ASSIGN AVAILABLE SLOTS
    for ((slot=0; slot<MAX_PARALLEL; slot++)); do
        free_slots+=("$slot")
    done

    # PID > SLOT MAPPING
    declare -A pid_to_slot

    while IFS= read -r line; do
        # IF NO FREE CPU SLOT
        # WAIT UNTIL ONE FINISHES
        while ((${#free_slots[@]} == 0)); do # FOR THE FIRST ITERATION, free_slots IS SUPPOSED TO BE FULL, THEREFORE THIS WHILE WILL NOT BE CAUGTH
            wait -n
            # FIND IDLING PID(s)
            for pid in "${!pid_to_slot[@]}"; do # ! SIGN > RETURNS KEYS, INSTEAD OF CORRESPONDING ARRAY ELEMENTS
                # kill -0; DOES NOT KILL THE PROCESS SET, INSTEAD SIMPLY CHECKS IF THE SET OF ID STILL ALIVE
                # EXAMPLE> (EXISTS > 0, NOT EXISTS > 1)
                if ! kill -0 "$pid" 2>/dev/null; then
                    # ONLY WHEN PID IS DEAD
                    local freed_slot="${pid_to_slot[$pid]}"
                    echo " [FREE ] PID=$pid RELEASED SLOT=$freed_slot"
                    free_slots+=($freed_slot) # RETRIEVE RESOURCE
                    unset pid_to_slot["$pid"] # DROP PID THAT WAS DETACHED
                fi
            done
        done

        # ---------------------------------------------
        # ALLOCATE FIRST AVAILABLE SLOTS
        # ---------------------------------------------
        slot="${free_slots[0]}"
        free_slots=("${free_slots[@]:1}")
        # "${free_slots[@}" : ALL ELEMENTS // ("${free_slots[@]:1}") WHERE ':1' TRIM THE ARRAY AND KEEP IT FROM THE ELEMENT '1'

        local cpu_start=$((slot*CORES_PER_JOB))

        IFS='|' read -r PTASKCODE TASKID OPTIONPATH PARAMETERPATH <<< "$line"
        # TASKID-WISE LOGGING
        # > SEPERATE OUTPUTS OF "process_bms()"
        process_bms "$line" "$update_time" "$cpu_start" "$REQUEST_DATE" \
                    > "${TASKID}_UPDATE.log" 2>&1 &
        # process_bms() IS SUPPOSED TO USE A CERTAIN COUNT OF CPU SET, THE SET WILL BE UNDER MANAGING WITH THE PIDS

        # STORE THE TASK PID THAT JUST SUBMITTED BACKGROUND (process_bms)
        pid=$!
        # SAVE THE PID WITHIN ASSOCIATIVE ARRAY 'pid_to_slot'
        pid_to_slot["$pid"]="$slot"

        echo " [START] PID=$pid SLOT=$slot CPU_START=$cpu_start JOB=$line"

    done < "$worklist"


    # ----------------------------------------
    # FINALISING : PID CLEAN-UP PHASE 
    # ----------------------------------------
    while ((${#pid_to_slot[@]} > 0)); do

        wait -n
        for pid in "${!pid_to_slot[@]}"; do
            if ! kill -0 "$pid"  2>/dev/null; then
                echo " [DONE ] PID=$pid SLOT=${pid_to_slot[$pid]}"
                unset pid_to_slot["$pid"]
            fi
        done
    done

    echo " [FINALISING] BMS RUL UPDATE COMPLETED"
}





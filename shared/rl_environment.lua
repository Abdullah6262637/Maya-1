-- Causal LLM Reinforcement Learning (RL) Environment Simulation in Lua
-- Calculates reward signals, penalties, and policy evaluations for agent training.

local RLEnvironment = {}

-- ----------------------------------------------------------------
-- Calculate reward score for a generated sequence of tokens
-- ----------------------------------------------------------------
function RLEnvironment.calculate_reward(generated_tokens, target_tokens)
    local score = 0.0
    local matches = 0
    local length = #generated_tokens
    
    if length == 0 then
        return -1.0 -- Severe penalty for empty generation
    end

    -- 1. Precision match against targets
    for i = 1, math.min(#generated_tokens, #target_tokens) do
        if generated_tokens[i] == target_tokens[i] then
            matches = matches + 1
            score = score + 0.1
        end
    end

    -- 2. Bonus for high matches ratio (token precision)
    local accuracy = matches / length
    if accuracy > 0.8 then
        score = score + 1.0
    elseif accuracy > 0.5 then
        score = score + 0.5
    end

    -- 3. Length penalty
    local penalty = RLEnvironment.length_penalty(length, 128)
    score = score - penalty

    return score
end

-- ----------------------------------------------------------------
-- Computes penalty factor for text length anomalies
-- ----------------------------------------------------------------
function RLEnvironment.length_penalty(current_length, target_length)
    local diff = math.abs(current_length - target_length)
    if diff > 50 then
        return 0.5 -- Excessive mismatch penalty
    elseif diff > 20 then
        return 0.2
    else
        return 0.0
    end
end

-- Module exporter
return RLEnvironment

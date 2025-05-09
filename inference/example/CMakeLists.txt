# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
function(add_llama_example name)
    set(tgt example-bl-llama-${name})
    add_executable(${tgt} e-${name}.cpp)
    target_link_libraries(${tgt} PRIVATE
        bl::llama
        ac-test-data::llama
        jalog::sinklib
    )
    set_target_properties(${tgt} PROPERTIES FOLDER example)
endfunction()

add_llama_example(basic)
add_llama_example(verify)

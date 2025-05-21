# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
function(add_server_example name)
    set(tgt example-bl-${name})
    add_executable(${tgt} e-${name}.cpp)
    target_link_libraries(${tgt} PRIVATE
        bl::llama-server
        ac-test-data::llama
        jalog::sinklib
    )
    set_target_properties(${tgt} PROPERTIES FOLDER example)
endfunction()

add_server_example(server-cli)

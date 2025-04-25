# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
if(NOT TARGET minja)
    CPMAddPackage(
        NAME minja
        GIT_REPOSITORY https://github.com/google/minja.git
        GIT_TAG dee1b8921ccdc51846080fda5299bae2b592d354
        SYSTEM TRUE
        DOWNLOAD_ONLY YES
    )

    add_library(minja INTERFACE)
    target_include_directories(minja INTERFACE ${minja_SOURCE_DIR}/include)
endif()

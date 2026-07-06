# OxC3(Oxsomi core 3), a general framework and toolset for cross-platform applications.
# Copyright (C) 2023 - 2026 Oxsomi / Nielsbishere (Niels Brunekreef)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://github.com/Oxsomi/core3/blob/main/LICENSE.
# Be aware that GPL3 requires closed source products to be GPL3 too if released to the public.
# To prevent this a separate license will have to be requested at contact@osomi.net for a premium;
# This is called dual licensing.

file(GLOB_RECURSE aiuHeaders "../../include/aiu/*.h")
file(GLOB_RECURSE aiuSharedHeaders "../../include/aiu/shared/*.h")
file(GLOB aiuSources "*.c")

add_library(
	OxC3_aiu
	STATIC
	${aiuHeaders}
	${aiuSharedHeaders}
	${aiuSources}
	CMakeLists.txt
)

source_group("Header Files (shared)" FILES ${aiuSharedHeaders})

set_target_properties(OxC3_aiu PROPERTIES FOLDER Oxsomi/aiu)
target_include_directories(OxC3_aiu PUBLIC "../../include")
target_link_libraries(OxC3_aiu PUBLIC OxC3_platforms OxC3_types_base OxC3_types_math OxC3_types_container)

if(EnableTests)

	file(GLOB aiuTests "test/*.c")

	add_executable(
		OxC3_aiu_test
		${aiuTests}
		CMakeLists.txt
	)

	set_target_properties(OxC3_aiu_test PROPERTIES FOLDER Oxsomi/aiu)
	target_link_libraries(OxC3_aiu_test PUBLIC OxC3_aiu OxC3_types_container_test_util)
	target_include_directories(OxC3_aiu_test PUBLIC "test")
	add_test(NAME OxC3_aiu_test COMMAND OxC3_aiu_test)

endif()

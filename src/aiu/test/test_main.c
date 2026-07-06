/* OxC3(Oxsomi core 3), a general framework and toolset for cross-platform applications.
*  Copyright (C) 2023 - 2026 Oxsomi / Nielsbishere (Niels Brunekreef)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see https://github.com/Oxsomi/core3/blob/main/LICENSE.
*  Be aware that GPL3 requires closed source products to be GPL3 too if released to the public.
*  To prevent this a separate license will have to be requested at contact@osomi.net for a premium;
*  This is called dual licensing.
*/

//aiu/test/test_main.c

#include "test_aiu_shared.h"
#include "platforms/platform.h"
#include "types/test/test.h"
#include "types/base/allocator.h"

Platform_defineEntrypoint() {

	Error err = Error_none();
	if (!Platform_create(Platform_argc, Platform_argv, Platform_getData(), NULL, true, &err)) {
		// Can't even set up the platform , hard fail
		Platform_return(1);
	}

	Test t = (Test) { .alloc = Platform_instance->alloc };

	Test_quant(&t);
	Test_gguf(&t);

	int result = Test_end(&t);
	Platform_cleanup();
	Platform_return(result);
}

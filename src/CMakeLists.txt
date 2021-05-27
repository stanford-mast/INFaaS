project(infaas-src)

add_subdirectory(metadata-store)

if(BUILD_ONLY_WORKER)
  message(STATUS "Building - worker")
  add_subdirectory(worker)
elseif(BUILD_ONLY_MD)
  message(STATUS "Building - metadata")
else(BUILD_ONLY_WORKER)
  message(STATUS "Building - all")
  add_subdirectory(worker)
  add_subdirectory(master)
  add_subdirectory(cli-tools)
endif(BUILD_ONLY_WORKER)


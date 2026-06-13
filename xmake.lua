add_rules("mode.debug", "mode.release")
set_policy("build.warning", true)
set_warnings("all")

add_requires("eigen", "openmp")

target("learn_mlp", function()
    set_kind("binary")
    set_languages("c17", "c++23")
    add_files("mlp.cpp")
    add_cxxflags("-march=native")
    add_packages("eigen", "openmp")
end)
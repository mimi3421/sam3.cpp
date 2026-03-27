add_executable(test_load test_load.cpp)
target_link_libraries(test_load PRIVATE sam3)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE sam3)

add_executable(test_vit test_vit.cpp)
target_link_libraries(test_vit PRIVATE sam3)

add_executable(test_phase5 test_phase5.cpp)
target_link_libraries(test_phase5 PRIVATE sam3)

add_executable(test_phase6 test_phase6.cpp)
target_link_libraries(test_phase6 PRIVATE sam3)

add_executable(test_phase3 test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE sam3)

add_executable(test_phase4 test_phase4.cpp)
target_link_libraries(test_phase4 PRIVATE sam3)

add_executable(test_phase7 test_phase7.cpp)
target_link_libraries(test_phase7 PRIVATE sam3)

add_executable(test_debug_encoder test_debug_encoder.cpp)
target_link_libraries(test_debug_encoder PRIVATE sam3)

add_executable(test_preprocess_match test_preprocess_match.cpp)
target_link_libraries(test_preprocess_match PRIVATE sam3)

add_executable(test_jpeg_decode test_jpeg_decode.cpp)
target_link_libraries(test_jpeg_decode PRIVATE sam3)

add_executable(test_geom_enc test_geom_enc.cpp)
target_link_libraries(test_geom_enc PRIVATE sam3)

add_executable(test_debug_fenc test_debug_fenc.cpp)
target_link_libraries(test_debug_fenc PRIVATE sam3)

add_executable(test_text_enc_dump test_text_enc_dump.cpp)
target_link_libraries(test_text_enc_dump PRIVATE sam3)

add_executable(test_e2e_pvs test_e2e_pvs.cpp)
target_link_libraries(test_e2e_pvs PRIVATE sam3)

add_executable(test_e2e_pcs test_e2e_pcs.cpp)
target_link_libraries(test_e2e_pcs PRIVATE sam3)

add_executable(test_text_enc_cat test_text_enc_cat.cpp)
target_link_libraries(test_text_enc_cat PRIVATE sam3)

add_executable(test_llama_pcs test_llama_pcs.cpp)
target_link_libraries(test_llama_pcs PRIVATE sam3)

add_executable(test_pvs_point test_pvs_point.cpp)
target_link_libraries(test_pvs_point PRIVATE sam3)

add_executable(test_video_2frame test_video_2frame.cpp)
target_link_libraries(test_video_2frame PRIVATE sam3)

add_executable(test_video_text test_video_text.cpp)
target_link_libraries(test_video_text PRIVATE sam3)

add_executable(test_metal_win test_metal_win.cpp)
target_link_libraries(test_metal_win PRIVATE sam3)

add_executable(test_metal_vit test_metal_vit.cpp)
target_link_libraries(test_metal_vit PRIVATE sam3)

add_executable(test_metal_conv_dw test_metal_conv_dw.cpp)
target_link_libraries(test_metal_conv_dw PRIVATE sam3)


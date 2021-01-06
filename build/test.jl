module test

A = zeros(Float64, 2, 3)

function test_function!(test)
    test .= 1
end

function test_function2!()
    test_function!(A)
    print(A)
end

end  # module test

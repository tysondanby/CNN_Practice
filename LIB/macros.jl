macro suppress_output(ex)
    return quote
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                $(esc(ex))
            end
        end
    end
end
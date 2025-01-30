using Gtk


function importnew()
    #GET DIR
    dir = open_dialog("Select Dataset Folder", action=GtkFileChooserAction.SELECT_FOLDER)
    if isdir(dir)
        # do something with dir
    end
    #VERIFY FOLDER STRUCTURE AND PRESENCE OF FILES
    #COPY IMAGES
    #COPY AUDIO
    #COPY OTHER DATA
end
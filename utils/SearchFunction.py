import pandas as pd


df = pd.read_csv('.\\weights\\image_faces_flags.csv')




def searchdb(faces, flags):

    if faces and flags:


        df['face_matches'] = df['faces'].apply(
            lambda x: sum(face in x for face in faces) if pd.notnull(x) else 0
        )
        

        df['flag_matches'] = df['flags'].apply(
            lambda x: sum(flag in x for flag in flags) if pd.notnull(x) else 0
        )
        
        matching_df = df[
            (df['face_matches'] > 0) & (df['flag_matches'] > 0)
        ]
        
        if not matching_df.empty:
            matching_df = matching_df.sort_values(by=['face_matches'], ascending=False)

            return matching_df['image_id'].tolist()  
        
        sorted_df = df.sort_values(by='face_matches', ascending=False)
        result = []
        for match_count in range(len(faces), 0, -1):  
            
            subset = sorted_df[sorted_df['face_matches'] == match_count]
            result.extend(subset['image_id'].tolist())
        
        return result  
    
    elif not faces and flags:
        df['flag_matches'] = df['flags'].apply(
            lambda x: sum(flag in x for flag in flags) if pd.notnull(x) else 0
        )
        sorted_df = df[
            (df['faces'].isnull() | (df['faces'] == ""))
        ].sort_values(by='flag_matches', ascending=False)
        
        result = []
        for match_count in range(len(flags), 0, -1):  
            subset = sorted_df[sorted_df['flag_matches'] == match_count]
            result.extend(subset['image_id'].tolist())
        
        return result 
    elif faces and not flags:
        df['face_matches'] = df['faces'].apply(
            lambda x: sum(face in x for face in faces) if pd.notnull(x) else 0
        )
        sorted_df = df[
            (df['flags'].isnull() | (df['flags'] == ""))
        ].sort_values(by='face_matches', ascending=False)
        
        result = []
        for match_count in range(len(faces), 0, -1): 
            subset = sorted_df[sorted_df['face_matches'] == match_count]
            result.extend(subset['image_id'].tolist())
        return result 
    
    return []


